"""Base class to facilitate buffer registration"""
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
from typing_extensions import override

from . import __version__
from .containers import TModule
from .utils import clear_cache

logger = logging.getLogger(__name__)
ComponentT = TypeVar("ComponentT", bound="Component")


class Call(NamedTuple):
    cmpt: Component
    fun: str
    kwargs: dict[str, Any]


class Step(NamedTuple):
    calls: list[Call]
    greedy: bool = False


class BindingOptim(NamedTuple):
    ancestry: set[tuple[Component, ...]]
    steps: list[Step]

    def merge_binding_optim(self) -> None:
        """Merge unfreezes, redundant steps, redundant calls in a step"""
        # merge unfreezes
        call_to_step: dict[
            tuple[str, str, frozenset[tuple[str, Any]]], int
        ] = {}
        for step_idx, step in enumerate(self.steps):
            dropped_call_indices: set[int] = set()
            for call_idx, call in enumerate(step.calls):
                if call.fun == "unfreeze":
                    key = (
                        type(call.cmpt).__name__,
                        call.fun,
                        frozenset(call.kwargs.items()),
                    )
                    if key not in call_to_step:
                        call_to_step[key] = step_idx
                    else:
                        if call not in self.steps[call_to_step[key]].calls:
                            self.steps[call_to_step[key]].calls.append(call)
                        if step_idx != call_to_step[key]:
                            dropped_call_indices.add(call_idx)

            # remove redundant calls
            calls = {
                (call.cmpt, call.fun, frozenset(call.kwargs.items())): None
                for call_idx, call in enumerate(step.calls)
                if call_idx not in dropped_call_indices
            }
            step.calls[:] = [
                Call(cmpt, fun, dict(kwargs)) for (cmpt, fun, kwargs) in calls
            ]

        # remove redundant or empty steps
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


class Component(TModule, abc.ABC):
    """Module that serves as a component in ProBound

    Attributes:
        _unfreezable:
            A Literal of all strings that can be passed to self.unfreeze().
        _cache_fun:
            A string representing the name of a function in a
            module's components whose output the module depends on.
        _blocking:
            A dictionary where each key is the name of the function being
            cached and the value is a set of all components that are
            currently waiting on that function's output.
        _caches:
            A dictionary where each key is the name of the function being
            cached and the value is a tuple of two optional elements:
            the pointer to the input and the cache of the output.
    """

    _unfreezable = Literal["all"]
    _cache_fun = "forward"

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name
        self._blocking: dict[str, set[Component]] = collections.defaultdict(
            set
        )
        self._caches: dict[str, tuple[int | None, Tensor | None]] = {}

    @override
    def __str__(self) -> str:
        return f"{self.name}{self.__class__.__name__}"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def save(
        self,
        checkpoint: torch.serialization.FILE_LIKE,
        flank_lengths: list[tuple[int, int]],
    ) -> None:
        """Save model to checkpoint file"""
        metadata = {
            "time": time.asctime(),
            "version": __version__,
            "flank_lengths": flank_lengths,
        }
        state_dict = self.state_dict()
        torch.save(
            {"state_dict": state_dict, "metadata": metadata}, checkpoint
        )

    def reload(
        self, checkpoint: torch.serialization.FILE_LIKE
    ) -> dict[str, Any]:
        """Load model from checkpoint file"""
        checkpoint_state = torch.load(checkpoint)
        checkpoint_state_dict = checkpoint_state["state_dict"]

        def get_attr(obj: Any, names: list[str]) -> Any:
            if len(names) == 1:
                return getattr(obj, names[0])
            return get_attr(getattr(obj, names[0]), names[1:])

        def set_attr(obj: Any, names: list[str], val: Any) -> None:
            if len(names) == 1:
                setattr(obj, names[0], val)
            else:
                set_attr(getattr(obj, names[0]), names[1:], val)

        # update symmetry buffers
        for key in list(checkpoint_state_dict.keys()):
            if "symmetry" not in key:
                continue
            checkpoint_param = checkpoint_state_dict[key]
            submod_names = key.split(".")
            set_attr(self, submod_names, checkpoint_param)

        # reshape convolution matrices
        for module in self.modules():
            if hasattr(module, "update_params") and callable(
                module.update_params
            ):
                module.update_params()  # type: ignore[unreachable]

        # reshape remaining tensors
        for key in list(checkpoint_state_dict.keys()):
            checkpoint_param = checkpoint_state_dict[key]
            submod_names = key.split(".")
            try:
                self_attr = get_attr(self, submod_names)
            except AttributeError:
                continue
            if self_attr.shape != checkpoint_param.shape:
                if isinstance(self_attr, torch.nn.Parameter):
                    checkpoint_param = torch.nn.Parameter(
                        checkpoint_param, requires_grad=self_attr.requires_grad
                    )
                set_attr(self, submod_names, checkpoint_param)

        self.load_state_dict(checkpoint_state_dict, strict=False)
        return cast(dict[str, Any], checkpoint_state["metadata"])

    @abc.abstractmethod
    def components(self) -> Iterator[Component]:
        """Iterator of child components"""

    def freeze(self) -> None:
        """Freeze all parameters"""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        if parameter == "all":
            for cmpt in self.components():
                cmpt.unfreeze("all")
        else:
            raise ValueError(
                f"{type(self).__name__} cannot unfreeze parameter {parameter}"
            )

    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        """Returns dict mapping Binding.key() to BindingOptim"""
        if ancestry is None:
            ancestry = tuple()
        if current_order is None:
            current_order = {}
        for cmpt in self.components():
            current_order = cmpt.optim_procedure(
                ancestry + (self,), current_order
            )
        return current_order

    def max_embedding_size(self) -> int:
        """Maximum number of bytes needed to encode a sequence"""
        max_sizes = [i.max_embedding_size() for i in self.components()]
        return max(max_sizes + [1])

    def _apply_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
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

    @classmethod
    def cache(
        cls, fun: Callable[[ComponentT, Tensor], Tensor]
    ) -> Callable[[ComponentT, Tensor], Tensor]:
        """Score batch-first tensor of sequences, cache if necessary"""

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


class Transform(Component):
    """Component that applies a transformation to a tensor

    See https://github.com/pytorch/pytorch/issues/45414"""

    @override
    @abc.abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        ...

    @override
    def __call__(self, seqs: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(seqs))

    def check_length_consistency(self) -> None:
        bindings = {m for m in self.modules() if isinstance(m, Binding)}
        for binding in bindings:
            binding.check_length_consistency()

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # transforms are applied on sequences from the same experiment
        #    and binding keys cannot re-occur in an experiment
        bindings = {m for m in self.modules() if isinstance(m, Binding)}
        if len(bindings) != len({m.key() for m in bindings}):
            raise ValueError(f"Non-unique Binding component found in {self}")
        return super().optim_procedure(ancestry, current_order)


class Spec(Component):
    """Stores experiment-independent parameters"""

    @override
    def components(self) -> Iterator[Component]:
        return iter(())

    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        return binding_optim


class Binding(Transform, abc.ABC):
    """Abstract base class for binding modes and binding cooperativity"""

    @abc.abstractmethod
    def key(self) -> tuple[Spec, ...]:
        """Tuple of Spec used for combining binding components for training"""

    @abc.abstractmethod
    def expected_sequence(self) -> Tensor:
        ...

    def expected_log_score(self) -> float:
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.expected_sequence())
            self.train(training)
            return out.item()

    @abc.abstractmethod
    def score_windows(self, seqs: Tensor) -> Tensor:
        ...
