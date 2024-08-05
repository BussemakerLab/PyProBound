"""Base classes for typing and sequential optimization procedure encoding.

Members are explicitly re-exported in pyprobound.
"""

from __future__ import annotations

import abc
import collections
import functools
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal, NamedTuple, TypeVar, cast

import torch
from torch import Tensor
from torch.nn.modules.module import _addindent
from typing_extensions import override

from . import __version__
from .utils import clear_cache

logger = logging.getLogger(__name__)
ComponentT = TypeVar("ComponentT", bound="Component")


class Call(NamedTuple):
    """A function to be called during optimization.

    Run as `getattr(cmpt, fun)(**kwargs)`.

    Attributes:
        cmpt: The component to be called.
        fun: The name of the function to be called.
        kwargs: Any keyword arguments that will be passed to the function.
    """

    cmpt: Component
    fun: str
    kwargs: dict[str, Any]


class Step(NamedTuple):
    """A series of calls performed in a single step before re-optimizing.

    Attributes:
        calls: The calls that will be performed together before re-optimizing.
        greedy: Whether to repeat the calls each time the loss improves.
    """

    calls: list[Call]
    greedy: bool = False


class BindingOptim(NamedTuple):
    """The sequential optimization steps taken to fit a Binding component.

    Attributes:
        ancestry: A set of tuples where each successive component is a child
            component of the previous component, from the root to the Binding
            component to be optimized; one Binding can occur multiple times.
        steps: The sequential optimization steps to fit a Binding component.
    """

    ancestry: set[tuple[Component, ...]]
    steps: list[Step]

    def merge_binding_optim(self) -> None:
        """Merge all redundant steps and redundant calls in a step."""
        # Merge unfreezes
        call_to_step: dict[
            tuple[str, str, frozenset[tuple[str, Any]]], int
        ] = {}
        for step_idx, step in enumerate(self.steps):
            dropped_call_indices: set[int] = set()
            for call_idx, call in enumerate(step.calls):
                if call.fun in ("unfreeze", "activity_heuristic", "freeze"):
                    key = (
                        type(call.cmpt).__name__,
                        call.fun,
                        frozenset(call.kwargs.items()),
                    )
                    if call.fun == "activity_heuristic":
                        key = ("", call.fun, frozenset())
                    if key[-1] == frozenset([("parameter", "spacing")]):
                        key = (
                            "PSAM",
                            call.fun,
                            frozenset([("parameter", "monomer")]),
                        )
                    if key not in call_to_step:
                        call_to_step[key] = step_idx
                    else:
                        if call not in self.steps[call_to_step[key]].calls:
                            self.steps[call_to_step[key]].calls.append(call)
                        if step_idx != call_to_step[key]:
                            dropped_call_indices.add(call_idx)

            # Remove redundant calls
            calls = {
                (call.cmpt, call.fun, frozenset(call.kwargs.items())): None
                for call_idx, call in enumerate(step.calls)
                if call_idx not in dropped_call_indices
            }
            step.calls[:] = [
                Call(cmpt, fun, dict(kwargs)) for (cmpt, fun, kwargs) in calls
            ]

        # Remove redundant or empty steps
        calls_set: set[
            frozenset[tuple[Component, str, frozenset[tuple[str, Any]]]]
        ] = set()
        dropped_step_indices: set[int] = set()
        for step_idx, step in enumerate(self.steps):
            calls_set_key = frozenset(
                (call.cmpt, call.fun, frozenset(call.kwargs.items()))
                for call in step.calls
            )
            if len(calls_set_key) == 0 or calls_set_key in calls_set:
                dropped_step_indices.add(step_idx)
            else:
                calls_set.add(calls_set_key)
        self.steps[:] = [
            step
            for step_idx, step in enumerate(self.steps)
            if step_idx not in dropped_step_indices
        ]

        # Move 'unfreeze all' to the end
        try:
            idx = self.steps.index(
                Step(
                    [
                        Call(
                            next(iter(self.ancestry))[0],
                            "unfreeze",
                            {"parameter": "all"},
                        )
                    ]
                )
            )
            step = self.steps.pop(idx)
            self.steps.append(step)
        except ValueError:
            pass


class Component(torch.nn.Module, abc.ABC):
    """Module that serves as a component in PyProBound.

    Includes functions for loading from a checkpoint, freezing or unfreezing
    parameters, and defining sequential optimization procedures.

    Attributes:
        unfreezable: All possible values that can be passed to unfreeze().
        _cache_fun: The name of a function in the module's child components
            that will be cached to avoid recomputation.
        _blocking: A mapping from the name of the cached function to the
            parent components waiting on that function's output.
        _caches: A mapping from the name of the cached function to a tuple of
            two optional elements, the input pointer and the output cache.
    """

    unfreezable = Literal["all"]
    _cache_fun = "forward"

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name
        self._blocking: dict[str, set[Component]] = collections.defaultdict(
            set
        )
        self._caches: dict[str, tuple[int | None, Tensor | None]] = {}

    @override
    def __repr__(self) -> str:
        num_components = 0
        for _ in self.components():
            num_components += 1
        if num_components == 0:
            return f"{type(self).__name__}()"
        return (
            f"{type(self).__name__}( [\n  "
            + "\n  ".join(
                _addindent(repr(i), 2) + "," for i in self.components()  # type: ignore[no-untyped-call]
            )
            + "\n] )"
        )

    @override
    def __str__(self) -> str:
        if self.name != "":
            return f"{type(self).__name__}-{self.name}"
        return self.__repr__()

    def save(
        self,
        checkpoint: torch.serialization.FILE_LIKE,
        flank_lengths: tuple[tuple[int, int], ...] = tuple(),
    ) -> None:
        """Saves the model to a file with "state_dict" and "metadata" fields.

        Args:
            checkpoint: The file where the model will be checkpointed to.
            flank_lengths: The (left_flank_length, right_flank_length) of each
                table represented by the model, written to the metadata field.
        """
        metadata = {
            "time": time.asctime(),
            "version": __version__,
            "flank_lengths": flank_lengths,
        }
        state_dict = self.state_dict()
        torch.save(
            {"state_dict": state_dict, "metadata": metadata}, checkpoint
        )

    def reload_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the model from a state dict.

        Args:
            state_dict: The state dict, usually returned by self.state_dict().
        """

        def get_attr(obj: Any, names: list[str]) -> Any:
            if len(names) == 1:
                return getattr(obj, names[0])
            return get_attr(getattr(obj, names[0]), names[1:])

        def set_attr(obj: Any, names: list[str], val: Any) -> None:
            if len(names) == 1:
                setattr(obj, names[0], val)
            else:
                set_attr(getattr(obj, names[0]), names[1:], val)

        # Update symmetry buffers
        for key in list(state_dict.keys()):
            if "symmetry" not in key:
                continue
            checkpoint_param = state_dict[key]
            submod_names = key.split(".")
            set_attr(self, submod_names, checkpoint_param)

        # Reshape convolution matrices
        for module in self.modules():
            if hasattr(module, "update_params") and callable(
                module.update_params
            ):
                module.update_params()

        # Reshape remaining tensors
        for key in list(state_dict.keys()):
            checkpoint_param = state_dict[key]
            submod_names = key.split(".")
            try:
                self_attr = get_attr(self, submod_names)
            except AttributeError:
                continue
            if isinstance(self_attr, torch.nn.Parameter):
                checkpoint_param = torch.nn.Parameter(
                    checkpoint_param, requires_grad=self_attr.requires_grad
                )
            set_attr(self, submod_names, checkpoint_param)

    def reload(
        self, checkpoint: torch.serialization.FILE_LIKE
    ) -> dict[str, Any]:
        """Loads the model from a checkpoint file.

        Args:
            checkpoint: The file where the model state_dict was written to.

        Returns:
            The metadata field of the checkpoint file.
        """
        checkpoint_state: dict[str, Any] = torch.load(
            checkpoint, weights_only=True
        )
        checkpoint_state_dict: dict[str, Any] = checkpoint_state["state_dict"]
        self.reload_from_state_dict(checkpoint_state_dict)
        return cast(dict[str, Any], checkpoint_state["metadata"])

    @abc.abstractmethod
    def components(self) -> Iterator[Component]:
        """Iterator of child components."""

    def max_embedding_size(self) -> int:
        """The maximum number of bytes needed to encode a sequence.

        Used for splitting calculations to avoid GPU limits on tensor sizes.
        """
        max_sizes = [i.max_embedding_size() for i in self.components()]
        return max(max_sizes + [1])

    def freeze(self) -> None:
        """Turns off gradient calculation for all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self, parameter: unfreezable = "all") -> None:
        """Turns on gradient calculation for the specified parameter.

        Args:
            parameter: Parameter to be unfrozen, defaults to all parameters.
        """
        if parameter == "all":
            for cmpt in self.components():
                cmpt.unfreeze("all")
        else:
            raise ValueError(
                f"{type(self).__name__} cannot unfreeze parameter {parameter}"
            )

    def check_length_consistency(self) -> None:
        """Checks that input lengths of Binding components are consistent.

        Raises:
            RuntimeError: There is an input mismatch between components.
        """
        bindings = {m for m in self.modules() if isinstance(m, Binding)}
        for binding in bindings:
            binding.check_length_consistency()

    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        """The sequential optimization procedure for all Binding components.

        The optimization procedure is generated recursively through iteration
        over the child components of each module. All Binding components with
        the same specification returned from `key()` are trained jointly.

        Args:
            ancestry: The parent components from the root for which the
                procedure is being generated to the current component.
            current_order: Mapping of Binding component specifications to the
                sequential optimization procedure for those Binding components.

        Returns:
            The `current_order` updated with the optimization of the current
            component's children.
        """
        if ancestry is None:
            ancestry = tuple()
        if current_order is None:
            current_order = {}
        for cmpt in self.components():
            current_order = cmpt.optim_procedure(
                ancestry + (self,), current_order
            )
        return current_order

    def _apply_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        """Directs the storage of intermediate results to avoid recomputation.

        Args:
            component: The parent component applying a block.
            cache_fun: The function whose output will be cached.
        """
        if component is not None and cache_fun is not None:
            logger.info(
                "Applying block of %s on %s.%s", component, self, cache_fun
            )
            self._blocking[cache_fun].add(component)
            logger.debug("%s._blocking=%s", self, self._blocking)

        for cmpt in self.components():
            # pylint: disable-next=protected-access
            cmpt._apply_block(self, self._cache_fun)

    def _release_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        """Releases intermediate results, called after output has been used.

        Args:
            component: The parent component releasing the block.
            cache_fun: The function whose output will be released.
        """
        if component is not None and cache_fun is not None:
            logger.info(
                "Releasing block of %s on %s.%s", component, self, cache_fun
            )
            self._blocking[cache_fun].discard(component)
            logger.debug("%s._blocking=%s", self, self._blocking)

            if len(self._blocking[cache_fun]) == 0:
                logger.info("Clearing cache of %s.%s", self, cache_fun)
                self._caches[cache_fun] = (None, None)
                logger.debug("%s._caches=%s", self, self._caches)
                clear_cache()

        for cmpt in self.components():
            # pylint: disable-next=protected-access
            cmpt._release_block(self, self._cache_fun)


class Transform(Component):
    """Component that applies a transformation to a tensor.

    Includes improved typing and caching outputs to avoid recomputation for
    transformations that appear multiple times in a loss module. See
    https://github.com/pytorch/pytorch/issues/45414 for typing information.
    """

    @override
    @abc.abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        """A transformation applied to a sequence tensor."""

    @override
    def __call__(self, seqs: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(seqs))

    @classmethod
    def cache(
        cls, fun: Callable[[ComponentT, Tensor], Tensor]
    ) -> Callable[[ComponentT, Tensor], Tensor]:
        """Decorator for a function to cache its output.

        The decorator must be applied to every function call whose output will
        be used in the cached function - generally all forward definitions.
        """

        @functools.wraps(fun)
        def cache_decorator(self: ComponentT, seqs: Tensor) -> Tensor:
            # pylint: disable=protected-access
            data_ptr = seqs.data_ptr()
            logger.info("Calling %s.%s(%s)", self, fun.__name__, data_ptr)
            logger.debug("%s._caches=%s", self, self._caches)
            ptr, output = self._caches.get(fun.__name__, (None, None))
            if output is not None:
                if ptr != data_ptr:
                    raise RuntimeError(
                        "Cached input pointer does not match current input"
                    )
                logger.info("Returning cache of %s.%s", self, fun.__name__)
                return output

            self._apply_block()

            logger.info("Calculating output of %s.%s", self, fun.__name__)
            output = fun(self, seqs)

            self._release_block()

            if len(self._blocking[fun.__name__]) > 0:
                logger.info("Caching output of %s.%s", self, fun.__name__)
                self._caches[fun.__name__] = (data_ptr, output)
                logger.debug("%s._caches=%s", self, self._caches)
            # pylint: enable=protected-access

            return output

        return cache_decorator


class Spec(Component):
    """A component that stores experiment-independent parameters.

    The forward implementation should be left to the experiment-specific
    implementation (either a Layer or Cooperativity component).
    """

    @override
    def components(self) -> Iterator[Component]:
        return iter(())

    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        """Updates a BindingOptim with the specification's optimization steps.

        Args:
            binding_optim: The parent BindingOptim to be updated.

        Returns:
            The updated BindingOptim.
        """
        return binding_optim


class Binding(Transform, abc.ABC):
    """Abstract base class for binding modes and binding cooperativity.

    Each Binding component links a specification storing experiment-independent
    parameters with the matching experiment and its specific parameters.
    """

    @abc.abstractmethod
    def key(self) -> tuple[Spec, ...]:
        """The specification of a Binding component.

        All Binding components with the same specification will be optimized
        together in the sequential optimization procedure.
        """

    @abc.abstractmethod
    def expected_sequence(self) -> Tensor:
        """Uninformative prior of input, used for calculating expectations."""

    def expected_log_score(self) -> float:
        """Calculates the expected log score."""
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.expected_sequence())
            self.train(training)
            return out.item()

    @abc.abstractmethod
    def score_windows(self, seqs: Tensor) -> Tensor:
        r"""Calculates the score of each window before summing over them.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            A tensor with the score of each window.
        """
