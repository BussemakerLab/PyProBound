"""Optimizer of LossModules.

Members are explicitly re-exported in pyprobound.
"""

import collections
import inspect
import io
import os
import sys
import tempfile
import timeit
import warnings
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Generic, TypeVar, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from .aggregate import Contribution
from .base import Call, Spec, Step
from .layers import Conv1d
from .loss import BaseLoss, Loss
from .mode import Mode
from .table import Batch, MultitaskLoader, Table
from .utils import clear_cache

STDOUT = cast(io.TextIOBase, sys.stdout)
POSINF = torch.tensor(float("inf"))
T = TypeVar("T", bound=Batch)

# TODO: update printing


def _file_not_empty(path: str | os.PathLike[str] | io.TextIOBase) -> bool:
    """Check if a file is exists and is not empty."""
    if isinstance(path, (str, os.PathLike)) and os.path.isfile(path):
        return os.stat(path).st_size > 0
    return False


class Optimizer(Generic[T]):
    """Optimizer of a BaseLoss."""

    def __init__(
        self,
        model: BaseLoss[T],
        train_tables: Sequence[Table[T]],
        val_tables: Sequence[Table[T]] | None = None,
        epochs: int = 200,
        patience: int = 10,
        greedy_threshold: float = 2e-4,
        batch_size: int | None = None,
        checkpoint: str | os.PathLike[str] = "model.pt",
        output: str | os.PathLike[str] | io.TextIOBase = STDOUT,
        device: str | None = None,
        sampler: type[Sampler[T]] | None = None,
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        optim_args: MutableMapping[str, Any] | None = None,
        sampler_args: MutableMapping[str, Any] | None = None,
    ) -> None:
        r"""Initializes the optimizer.

        Args:
            model: The loss module to be optimized.
            train_tables: The tables to be trained against.
            val_tables: The tables to be validated against for early stopping.
            epochs: The maximum number of epochs taken until convergence.
            patience: The maximum number of epochs taken without improvement of
                the train loss (or validation loss if available).
            greedy_threshold: The change in loss necessary to accept a Step
                with attribute `greedy` set to True.
            batch_size: The number of sequences used to optimize the model at a
                time.
            checkpoint: The file where the model will be checkpointed to.
            output: The file where the optimization output will be written to.
            device: The device on which to perform optimization.
            sampler: The sampler used when creating the dataloader.
            optimizer: The optimizer used for optimization.
            optim_args: Parameters passed to the optimizer.
                (Defaults to `{"line_search_fn":"strong_wolfe"}` if available).
            sampler_args: Parameters passed to the sampler.
        """

        if len(list(model.components())) != len(train_tables):
            raise ValueError(
                "Number of experiments and training datasets must be equal"
            )
        self.train_tables = train_tables

        # Fill in defaults
        if optim_args is None:
            optim_args = {}
        if sampler_args is None:
            sampler_args = {}

        # Store count tables
        self._tables: tuple[tuple[Table[T], ...], ...] = tuple(
            zip(train_tables)
        )
        if val_tables is not None:
            if len(list(model.components())) != len(val_tables):
                raise ValueError(
                    "Number of models and validation datasets must be equal"
                )
            self._tables = tuple(zip(train_tables, val_tables))

        # Make DataLoader objects for training and validation data
        self.train_dataloader = MultitaskLoader(
            [
                DataLoader(
                    table,
                    batch_size=(
                        len(table) if batch_size is None else batch_size
                    ),
                    sampler=(
                        sampler(table, **sampler_args)
                        if sampler is not None
                        else None
                    ),
                )
                for table in self.train_tables
            ]
        )
        self.val_dataloader = None
        if val_tables is not None:
            self.val_dataloader = MultitaskLoader(
                [
                    DataLoader(
                        table,
                        batch_size=(
                            len(table) if batch_size is None else batch_size
                        ),
                        sampler=(
                            sampler(table, **sampler_args)
                            if sampler is not None
                            else None
                        ),
                    )
                    for table in val_tables
                ]
            )

        # Check output and checkpoint files for current contents
        if _file_not_empty(output):
            warnings.warn(f"Output file {output} is not empty")
        if _file_not_empty(checkpoint):
            warnings.warn(f"Checkpoint file {checkpoint} is not empty")

        # Get device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cpu":
            for module in model.modules():
                if isinstance(module, Conv1d) and module.one_hot:
                    warnings.warn("one_hot is extremely slow on CPU")
        if device != "cpu":
            for module in model.modules():
                if isinstance(module, Conv1d) and not module.one_hot:
                    warnings.warn("dense is extremely slow on GPU")

        # Store class attributes
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.patience = patience
        self.checkpoint = checkpoint
        self.output = output
        self.optimizer_type = optimizer
        self.optim_args = optim_args
        self.optimizer: torch.optim.Optimizer | None = None
        self.greedy_threshold = greedy_threshold
        if (
            "line_search_fn"
            in inspect.signature(self.optimizer_type).parameters
            and "line_search_fn" not in self.optim_args
        ):
            self.optim_args["line_search_fn"] = "strong_wolfe"

        self.check_length_consistency()

    def get_parameter_string(self) -> str:
        """A prettified representation of the parameters of the model."""

        def param_str(param: Tensor, n_tabs: int = 3) -> str:
            old_pad = " " * 8
            new_pad = "\t" * (n_tabs) + old_pad
            return ("\n" + new_pad).join(str(param).split("\n"))

        torch.set_printoptions(threshold=10)  # type: ignore[no-untyped-call]
        out = []
        psam: collections.defaultdict[
            tuple[str, bool], list[torch.nn.Parameter]
        ] = collections.defaultdict(list)
        coop_posbias: collections.defaultdict[
            str, list[torch.nn.Parameter]
        ] = collections.defaultdict(list)
        for name, param in self.model.named_parameters():
            if "layer_spec.betas" in name:
                base, key = name.rsplit(".", 1)
                psam[(base, len(key.split("-")) == 3)].append(param)
            elif "log_posbias." in name:
                base, key = name.rsplit(".", 1)
                coop_posbias[base].append(param)
            else:
                out.append(f"\t\t\t\t{name} grad={param.requires_grad}")
                if param.numel() <= 1:
                    out[-1] += f" {param.detach()}"
                else:
                    out.append(f"\t\t\t\t\t{param_str(param.detach())}")

        torch.set_printoptions(threshold=100)  # type: ignore[no-untyped-call]
        for (key, interaction), val in psam.items():
            name = f"{key}-{'pairwise' if interaction else 'monomer'}"
            flat_param = torch.stack(cast(list[torch.Tensor], val))
            out.append(f"\t\t\t\t{name} grad={flat_param.requires_grad}")
            out.append(f"\t\t\t\t\t{param_str(flat_param.detach())}")
        for key, val in coop_posbias.items():
            flat_param = torch.cat(
                cast(list[torch.Tensor], [i.flatten() for i in val])
            )
            out.append(f"\t\t\t\t{key} grad={flat_param.requires_grad}")
            out.append(f"\t\t\t\t\t{param_str(flat_param.detach())}")

        return "\n".join(out)

    def get_setup_string(self) -> str:
        """A description of the current model and tables."""
        out = ["\n### Tables:"]
        for ct_idx, ct in enumerate(self.train_tables):
            out.extend([f"\tTable: {ct_idx}", ct.get_setup_string()])

        return self.model.get_setup_string() + "\n" + "\n".join(out)

    def get_train_sequential(
        self, order: Iterable[Iterable[tuple[Spec, ...]]] | None = None
    ) -> str:
        """A description of the sequential optimization steps to be taken.

        Args:
            order: An iterable encoding the training order, where each element
                is an iterable of binding keys to be trained simultaneously.
        """
        out: list[str] = []
        optim_procedure = self.model.optim_procedure()
        if order is None:
            order = [[i] for i in optim_procedure.keys()]
        for keys in order:
            # Merge optimization of all modes
            binding_optims = [optim_procedure[key] for key in keys]
            binding_optim = binding_optims[0]
            for alt_optim in binding_optims[1:]:
                for ancestry in alt_optim.ancestry:
                    binding_optim.ancestry.add(ancestry)
                binding_optim.steps.extend(alt_optim.steps)
            binding_optim.merge_binding_optim()
            key_str = "### Binding: " + " and ".join(
                "-".join(str(i) for i in key) for key in keys
            )

            if len(out) == 0:
                out.append(key_str)
            else:
                out.append("\n" + key_str)
            for step_idx, step in enumerate(binding_optim.steps):
                out.append(f"\tStep {step_idx}")
                for call in step.calls:
                    parameters = ", ".join(
                        f"{k}={v}" for k, v in call.kwargs.items()
                    )
                    out.append(f"\t\t{call.cmpt}.{call.fun}({parameters})")
        return "\n".join(out)

    def print(
        self, *objects: Any, sep: str = " ", mode: str = "at", end: str = "\n"
    ) -> None:
        """Prints to self.output."""
        handle: io.TextIOBase
        try:
            if isinstance(self.output, io.TextIOBase):
                handle = self.output
            else:
                handle = cast(
                    io.TextIOBase,
                    # pylint: disable-next=consider-using-with
                    open(self.output, mode, encoding="utf-8"),
                )
            print(*objects, sep=sep, end=end, file=handle, flush=True)
        finally:
            if handle not in (
                cast(io.TextIOBase, sys.stdout),
                cast(io.TextIOBase, sys.stderr),
            ):
                handle.close()

    def save(self, checkpoint: torch.serialization.FILE_LIKE) -> None:
        """Saves the model to a file with "state_dict", "metadata" fields.

        Args:
            checkpoint: The file where the model will be checkpointed to.
        """
        self.model.save(
            checkpoint,
            flank_lengths=tuple(
                (ct.left_flank_length, ct.right_flank_length)
                for ct in self.train_tables
            ),
        )

    def reload(
        self, checkpoint: torch.serialization.FILE_LIKE | None = None
    ) -> dict[str, Any]:
        """Loads the model from a checkpoint file.

        Args:
            checkpoint: The file where the model state_dict was written to.
        """
        if checkpoint is None:
            checkpoint = self.checkpoint
        metadata = self.model.reload(checkpoint)
        for (left, right), tables in zip(
            metadata["flank_lengths"], self._tables
        ):
            for ct in tables:
                ct.set_flank_length(left, right)
        return metadata

    def check_length_consistency(self) -> None:
        """Checks that input lengths of Binding components are consistent.

        Raises:
            RuntimeError: There is an input mismatch between components.
        """
        for cmpt in self.model.components():
            cmpt.check_length_consistency()
        for cmpt, tables in zip(self.model.components(), self._tables):
            input_shapes = {table.input_shape for table in tables}
            if len(input_shapes) != 1:
                raise RuntimeError("Count table sequence shapes do not match")
            input_shape = input_shapes.pop()
            min_input_length = min(table.min_read_length for table in tables)
            max_input_length = min(table.max_read_length for table in tables)
            for bmd in cmpt.modules():
                if isinstance(bmd, Mode):
                    if bmd.input_shape != input_shape:
                        raise RuntimeError(
                            f"Expected input_shape={input_shape}"
                            f", found {bmd.input_shape}"
                        )
                    if bmd.min_input_length < min_input_length:
                        raise RuntimeError(
                            f"Expected min_input_length={min_input_length}"
                            f", found {bmd.min_input_length}"
                        )
                    if bmd.max_input_length > max_input_length:
                        raise RuntimeError(
                            f"Expected min_input_length={max_input_length}"
                            f", found {bmd.max_input_length}"
                        )

    def run_one_epoch(
        self, train: bool, dataloader: MultitaskLoader[T]
    ) -> Loss:
        """Runs one training epoch.

        Args:
            train: Whether to take an optimization step before returning Loss.
            dataloader: A multi-dataset loader.

        Returns:
            The loss calculated from the provided dataloader.
        """
        if self.optimizer is None:
            raise RuntimeError("Cannot optimize with uninitialized optimizer")

        neglogliks: list[Tensor] = []
        regularizations: list[Tensor] = []

        for batch in dataloader:
            # Take a step
            if train:
                self.model.train()

                def closure() -> float:
                    """Used to recompute loss."""
                    cast(torch.optim.Optimizer, self.optimizer).zero_grad()
                    # pylint: disable-next=cell-var-from-loop
                    loss = self.model(batch)
                    negloglik = loss.negloglik + loss.regularization
                    negloglik.backward()  # type: ignore[no-untyped-call]
                    return cast(float, negloglik)

                if (
                    "closure"
                    in inspect.signature(self.optimizer.step).parameters
                ):
                    self.optimizer.step(closure)
                else:
                    closure()
                    self.optimizer.step()

            # Calculate loss
            with torch.inference_mode():
                self.model.eval()
                nll, reg = self.model(batch)
                neglogliks.append(nll.detach().cpu())
                regularizations.append(reg.detach().cpu())

        return Loss(
            sum(neglogliks, start=torch.tensor(0.0)) / len(neglogliks),
            sum(regularizations, start=torch.tensor(0.0))
            / len(regularizations),
        )

    def train_until_convergence(
        self,
        checkpoint: torch.serialization.FILE_LIKE | None = None,
        best_loss: Tensor = POSINF,
    ) -> Tensor:
        """Repeat optimization steps until the loss stops improving.

        Args:
            checkpoint: The file where the model will be checkpointed to.
            best_loss: The previous loss used to determine if loss improved.

        Returns:
            The best loss calculated during the optimization procedure.
        """

        if checkpoint is None:
            checkpoint = self.checkpoint

        while True:
            # Reset optimizer
            self.optimizer = self.optimizer_type(
                [p for p in self.model.parameters() if p.requires_grad],
                **self.optim_args,
            )

            # Store initial state
            self.print(self.get_parameter_string())
            original_parameters = torch.cat(
                [
                    p.detach().flatten()
                    for p in self.model.parameters()
                    if p.requires_grad
                ]
            )
            patience_count = self.patience
            terminate_while = True  # If first step terminates, don't restart

            for epoch in range(self.epochs):
                start_time = timeit.default_timer()

                # Take a step
                train_nll, regularization = self.run_one_epoch(
                    True, self.train_dataloader
                )
                new_parameters = torch.cat(
                    [
                        p.detach().flatten()
                        for p in self.model.parameters()
                        if p.requires_grad
                    ]
                )
                distance = (
                    (new_parameters - original_parameters)
                    .square()
                    .sum()
                    .sqrt()
                )

                # Get validation loss
                if self.val_dataloader is not None:
                    with torch.inference_mode():
                        curr_nll, _ = self.run_one_epoch(
                            False, self.val_dataloader
                        )
                else:
                    curr_nll = train_nll

                # Save model if loss decreased
                original_parameters = new_parameters
                if (curr_nll + regularization) < best_loss:
                    self.print("\t\t\tLoss decreased")
                    self.save(checkpoint)
                    best_loss = curr_nll + regularization
                    patience_count = self.patience
                else:
                    patience_count -= 1

                # Print loss
                elapsed = float(timeit.default_timer() - start_time)
                self.print(
                    f"\t\t\tEpoch {epoch} took {elapsed:.2f}s"
                    + f" NLL: {train_nll:.10f} Reg.: {regularization:.10f}"
                    + (
                        f" Val: {curr_nll:.10f}"
                        if self.val_dataloader is not None
                        else ""
                    )
                    + f" Distance: {distance:.10f} Patience: {patience_count}"
                )

                # Progress in loops
                if (
                    distance == 0
                    or patience_count <= 0
                    or epoch == self.epochs - 1
                ):
                    # Terminate if reached minimum or ran out of patience
                    terminate_while = True
                    break
                if torch.isnan(curr_nll + regularization):
                    # Whether to restart depends on if progress has been made
                    break
                # Keep going
                terminate_while = False

            if terminate_while:
                break

        # Reload best model
        self.reload(checkpoint)
        self.print(self.get_parameter_string(), "\n")
        clear_cache()

        return best_loss

    def train_simultaneous(self, best_loss: Tensor = POSINF) -> Tensor:
        """Train all parameters in a model simultaneously.

        Args:
            best_loss: The previous loss used to determine if loss improved.

        Returns:
            The best loss calculated during the optimization procedure.
        """
        self.save(self.checkpoint)
        return self.train_until_convergence(best_loss=best_loss)

    def run_step(self, step: Step) -> None:
        """Run all calls specified in a step.

        Groups all `update_read_length` calls to be dispatched separately.
        """
        flank_calls: list[Call] = []

        for call_idx, call in enumerate(step.calls):
            # Print call
            pad = "\t" if call_idx == 0 else "\t\t"
            parameters = ", ".join(f"{k}={v}" for k, v in call.kwargs.items())
            self.print(f"{pad}{call.cmpt}.{call.fun}({parameters})")

            # Run call
            if call.fun == "update_read_length":
                flank_calls.append(call)
            else:
                getattr(call.cmpt, call.fun)(**call.kwargs)

        if len(flank_calls) > 0:
            self.update_read_length(flank_calls)

    def greedy_search(self, step: Step, best_loss: Tensor = POSINF) -> Tensor:
        """Run an optimization step repeatedly until the loss stops improving.

        Args:
            step: The step to be repeated until the loss stops improving.
            best_loss: The previous loss used to determine if loss improved.

        Returns:
            The best loss calculated during the optimization procedure.
        """
        greedy_fd, greedy_checkpoint = None, None

        contributions = [
            m for m in self.model.modules() if isinstance(m, Contribution)
        ]

        try:
            greedy_fd, greedy_checkpoint = tempfile.mkstemp()
            while True:
                # Get original contributions
                old_log_contributions = [
                    ctrb.expected_log_contribution() for ctrb in contributions
                ]

                # Update binding model
                try:
                    self.run_step(step)
                except ValueError as e:
                    self.print(f"\tGreedy search terminated: {e}")
                    self.reload(self.checkpoint)
                    break

                # Update activities to keep contributions constant
                for old_log_ctrb, ctrb in zip(
                    old_log_contributions, contributions
                ):
                    ctrb.set_contribution(old_log_ctrb)

                # Run fit
                self.save(greedy_checkpoint)
                if not any(p.requires_grad for p in self.model.parameters()):
                    nll, reg = self.run_one_epoch(False, self.train_dataloader)
                    loss = nll + reg
                else:
                    loss = self.train_until_convergence(
                        checkpoint=greedy_checkpoint, best_loss=best_loss
                    )

                # Retain update if loss improved
                if loss < (best_loss - self.greedy_threshold):
                    self.print("\tUpdate accepted")
                    best_loss = loss
                    self.save(self.checkpoint)
                else:
                    self.print("\tUpdate rejected")
                    self.reload(self.checkpoint)
                    break

        finally:
            if greedy_fd is not None:
                os.close(greedy_fd)
            if greedy_checkpoint is not None:
                os.remove(greedy_checkpoint)

        return best_loss

    def update_read_length(self, calls: list[Call]) -> None:
        """Combines all `update_read_length` calls across count tables."""

        # Get the count table index of each Mode
        bmd_to_ct_idx: dict[Mode, int] = {}
        for cmpt_idx, cmpt in enumerate(self.model.components()):
            for bmd in cmpt.modules():
                if isinstance(bmd, Mode):
                    bmd_to_ct_idx[bmd] = cmpt_idx

        # Order the calls into each count table
        ordered_kwargs: list[list[dict[str, Any]]] = [
            [] for _ in enumerate(self._tables)
        ]
        for call in calls:
            if not isinstance(call.cmpt, Mode):
                raise RuntimeError(
                    "update_read_length expected Mode"
                    f", found {type(call.cmpt).__name__} instead"
                )
            ordered_kwargs[bmd_to_ct_idx[call.cmpt]].append(call.kwargs)

        # Update count tables and binding modes
        for kwarg_list, cmpt, tables in zip(
            ordered_kwargs, self.model.components(), self._tables
        ):
            if len(kwarg_list) == 0:
                continue
            if len({frozenset(kwargs.items()) for kwargs in kwarg_list}) != 1:
                raise RuntimeError(
                    f"update_read_length calls {kwarg_list} are inconsistent"
                )
            kwargs = kwarg_list[0]
            for ct in tables:
                ct.set_flank_length(
                    left=ct.left_flank_length + kwargs.get("left_shift", 0),
                    right=ct.right_flank_length + kwargs.get("right_shift", 0),
                )
            for bmd in cmpt.modules():
                if isinstance(bmd, Mode):
                    bmd.update_read_length(**kwargs)

        self.check_length_consistency()

    def train_sequential(
        self,
        maintain_loss: bool = True,
        order: Iterable[Iterable[tuple[Spec, ...]]] | None = None,
    ) -> Tensor:
        """Train parameters according to the sequential optimization procedure.

        Args:
            maintain_loss: Whether to keep the `best_loss` from one step to the
                next when determining if the loss has improved.
            order: An iterable encoding the training order, where each element
                is an iterable of binding keys to be trained simultaneously.

        Returns:
            The best loss calculated during the optimization procedure.
        """

        self.print(self.get_setup_string())
        self.save(self.checkpoint)
        optim_procedure = self.model.optim_procedure()
        if order is None:
            order = [[i] for i in optim_procedure.keys()]
        for binding_idx, keys in enumerate(order):
            # Merge optimization of all modes
            binding_optims = [optim_procedure[key] for key in keys]
            binding_optim = binding_optims[0]
            for alt_optim in binding_optims[1:]:
                for ancestry in alt_optim.ancestry:
                    binding_optim.ancestry.add(ancestry)
                binding_optim.steps.extend(alt_optim.steps)
            binding_optim.merge_binding_optim()
            key_str = " and ".join(
                "-".join(str(i) for i in key) for key in keys
            )

            self.print(f"\n### Training Mode {binding_idx}: {key_str}")
            for ancestors in binding_optim.ancestry:
                self.print(f"\t{' → '.join(str(i) for i in ancestors)}")

            best_loss = POSINF
            self.save(self.checkpoint)
            for step_idx, step in enumerate(binding_optim.steps):
                self.print(f"\t{step_idx}.", end="")

                # Check if greedy search
                if step.greedy:
                    best_loss = self.greedy_search(step, best_loss=best_loss)
                    if not maintain_loss:
                        best_loss = POSINF
                    continue

                # Run calls
                self.run_step(step)
                self.save(self.checkpoint)

                # Don't optimize if no parameters can be trained
                if not any(p.requires_grad for p in self.model.parameters()):
                    continue

                # Optimize
                best_loss = self.train_until_convergence(best_loss=best_loss)
                if not maintain_loss:
                    best_loss = POSINF

        return best_loss
