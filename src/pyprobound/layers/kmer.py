r"""Nonparametric k-mer scoring layer.

Members are explicitly re-exported in pyprobound.layers.

A k-mer model replaces the additive PSAM of :class:`~pyprobound.layers.PSAM`
with a lookup table holding an independent :math:`-\Delta\Delta G_\phi/RT`
for each of the :math:`|A|^k` k-mers, so it can represent any interaction
within its footprint. Like :class:`~pyprobound.layers.Conv1d` the table is
scored over every sliding window of the input, but on one output channel: for
double-stranded data `rc_symmetry` ties each k-mer to its reverse complement,
which spans the same models as scoring both strands would and needs half the
degrees of freedom.
"""

import math
import warnings
from collections.abc import Iterator
from typing import Any, Literal, TypeVar, cast

import torch
from torch import Tensor
from typing_extensions import Self, override

from ..alphabets import Alphabet
from ..base import BindingOptim, Call, Step
from ..table import CountBatch, Table
from .layer import Layer, LayerSpec
from .psam import PSAM

T = TypeVar("T", int, Tensor)

# Fewer than this many windows per k-mer and the table is not estimable
_MIN_WINDOWS_PER_KMER = 50
# get_pwm: refinement passes, the column depth to keep, and the prior count
_REFINEMENTS = 5
_MIN_SUPPORT = 0.5
_PSEUDO = 1e-3


def _reverse_complement(table: Tensor, kmer_length: int) -> Tensor:
    r"""Reverse-complements the trailing k axes of a k-mer table.

    Assumes, as :class:`~pyprobound.alphabets.Alphabet` does, that the
    complement mapping is ``dict(zip(a, reversed(a)))``.
    """
    lead = table.ndim - kmer_length
    axes = tuple(range(lead, table.ndim))
    permutation = tuple(range(lead)) + tuple(reversed(axes))
    return table.permute(permutation).flip(axes)


class Kmers(LayerSpec):
    r"""A lookup table of :math:`-\Delta\Delta G_\phi/RT` for every k-mer.

    Attributes:
        kmers (Tensor): The score of every k-mer, of shape
            :math:`(\text{in_channels},)\times k`.
    """

    unfreezable = Literal[LayerSpec.unfreezable, "kmers"]

    def __init__(
        self,
        alphabet: Alphabet,
        kmer_length: int,
        rc_symmetry: bool | None = None,
        normalize: bool = True,
        backoff_length: int | None = None,
        init_from_counts: bool = True,
        name: str = "",
    ) -> None:
        r"""Initializes the k-mer model.

        Args:
            alphabet: The alphabet used to encode sequences into tensors.
            kmer_length: The k-mer length used for scoring.
            rc_symmetry: Whether to tie every k-mer to its reverse complement,
                defaulting to `alphabet.complement`. Halves the DOF.
            normalize: Whether to mean-center the table, removing its exact
                degeneracy against the enclosing `log_activity`.
            backoff_length: The contained k-mer that `get_dirichlet` shrinks
                towards, in :math:`[1, k-1]`, defaulting to :math:`k-1`. Set it
                to the footprint of the site; see `get_backoff`. Set the loss's
                `pseudocount` to 0 to apply no prior at all.
            init_from_counts: Whether to initialize from the observed k-mer
                enrichment on first attachment to a table. Without it the fit
                can stall outright; see `init_from_counts_`.
            name: A string used to describe the k-mer model.
        """
        if kmer_length < 1:
            raise ValueError(f"kmer_length={kmer_length} must be positive")
        if rc_symmetry is None:
            rc_symmetry = alphabet.complement
        if rc_symmetry and not alphabet.complement:
            raise ValueError("rc_symmetry needs an alphabet with a complement")
        if backoff_length is None:
            backoff_length = max(kmer_length - 1, 1)
        if not 1 <= backoff_length <= max(kmer_length - 1, 1):
            raise ValueError(f"backoff_length={backoff_length} out of range")

        super().__init__(
            out_channels=1, in_channels=len(alphabet.alphabet), name=name
        )
        self._layers: set[KmerScoring]  # type: ignore[assignment]
        self.alphabet = alphabet
        self._kmer_length = kmer_length
        self.rc_symmetry = rc_symmetry
        self.normalize = normalize
        self.backoff_length = backoff_length
        self.init_from_counts = init_from_counts
        self.counts_initialized = False

        self.kmers = torch.nn.Parameter(
            torch.zeros(size=(self.in_channels,) * kmer_length)
        )
        bound = math.sqrt(1 / (self.in_channels * self.kmer_length))
        torch.nn.init.uniform_(self.kmers, -bound, bound)

    @property
    def kmer_length(self) -> int:
        """The k-mer length used for scoring."""
        return self._kmer_length

    @override
    def __repr__(self) -> str:
        args = [f"kmer_length={self.kmer_length}"]
        if self.rc_symmetry:
            args.append("rc_symmetry=True")
        return f"{type(self).__name__}({', '.join(args)})"

    @override
    def components(self) -> Iterator[Any]:
        return iter(())

    def init_from_counts_(self, table: CountBatch) -> None:
        r"""Initializes the table to the observed log enrichment of each k-mer.

        A uniformly random table scores every window alike, so the likelihood
        is nearly flat and a line-search optimizer can fail to take any step at
        all. Windows containing a padding character are skipped.
        """
        seqs = table.seqs
        if seqs.ndim != 2:
            return  # probability-encoded input has no k-mer counts
        target = table.target
        if target.ndim != 2 or target.shape[1] < 2:
            return  # need at least an input and a selected round

        n_alpha = self.in_channels
        windows = seqs.unfold(-1, self.kmer_length, 1)
        valid = (windows < n_alpha).all(dim=-1)
        powers = n_alpha ** torch.arange(
            self.kmer_length - 1, -1, -1, device=seqs.device
        )
        index = (windows * powers).sum(dim=-1)[valid]
        rows = (
            torch.arange(len(seqs), device=seqs.device)
            .unsqueeze(1)
            .expand_as(valid)[valid]
        )

        n_kmers = n_alpha**self.kmer_length
        counts = [
            torch.bincount(
                index,
                weights=target[rows, column].to(torch.float64),
                minlength=n_kmers,
            )
            for column in (0, -1)
        ]
        enrichment = torch.log((counts[1] + 1.0) / (counts[0] + 1.0))
        enrichment = enrichment - enrichment.mean()
        with torch.no_grad():
            self.kmers.copy_(
                enrichment.reshape((n_alpha,) * self.kmer_length).to(
                    self.kmers.dtype
                )
            )
        self.counts_initialized = True

    def get_symmetrized(self) -> Tensor:
        r"""The table after applying `rc_symmetry`.

        The likelihood sees `kmers` only through this, so regularization must
        be computed on it rather than on `kmers` directly.
        """
        kmers: Tensor = self.kmers
        if self.rc_symmetry:
            kmers = (kmers + _reverse_complement(kmers, self.kmer_length)) / 2
        return kmers

    def get_table(self) -> Tensor:
        r"""The k-mer scores used for scoring, of shape
        :math:`(\text{out_channels},)+(\text{in_channels},)\times k`.
        """
        kmers = self.get_symmetrized()
        if self.normalize:
            kmers = kmers - kmers.mean()
        return kmers.unsqueeze(0)

    def get_padded_table(self) -> Tensor:
        r"""`get_table` extended with the alphabet's three padding indices.

        As in :meth:`~pyprobound.layers.Conv1d.score_dense`, ' ' maps to
        :math:`-\infty`. A k-mer score is not a sum of per-position terms, so
        '*' and '-' both map to the mean over that axis.
        """
        table = self.get_table()
        for axis in range(1, self.kmer_length + 1):
            mean = table.mean(dim=axis, keepdim=True)
            neginf = torch.full_like(mean, float("-inf"))
            table = torch.cat((table, neginf, mean, mean), dim=axis)
        return table

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length - self.kmer_length + 1

    @override
    def in_len(self, length: T, mode: Literal["min", "max"] = "max") -> T:
        del mode
        return length + self.kmer_length - 1

    @override
    def unfreeze(self, parameter: "Kmers.unfreezable" = "all") -> None:
        if parameter in ("kmers", "all"):
            self.kmers.requires_grad_()
        if parameter != "kmers":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        binding_optim.steps.append(
            Step([Call(self, "unfreeze", {"parameter": "kmers"})])
        )
        binding_optim.merge_binding_optim()
        return binding_optim

    def get_dirichlet(self) -> Tensor:
        """A fixed-variance Gaussian prior on the residual of `get_backoff`."""
        kmers = self.get_symmetrized()
        return -0.5 * torch.square(kmers - self.get_backoff()).sum()

    def get_backoff(self, backoff_length: int | None = None) -> Tensor:
        r"""What the contained :math:`m`-mers predict for each k-mer.

        With :math:`E_j` averaging over position :math:`j` and
        :math:`P_W = \prod_{j \notin W} E_j` projecting onto functions of a
        contiguous window :math:`W`, this returns
        :math:`\bigl(I - \prod_o (I - P_{W_o})\bigr)\beta` over the
        :math:`k-m+1` windows of length :math:`m` = `backoff_length`. At
        :math:`m=k-1` it is :math:`E_k + E_1 - E_1 E_k`, whose residual is the
        pure first-to-last interaction; at :math:`m=1` it is the closest PSAM.
        """
        kmers = self.get_symmetrized()
        length = (
            self.backoff_length if backoff_length is None else backoff_length
        )
        if not 1 <= length < self.kmer_length:
            raise ValueError(
                f"backoff_length={length} must be in"
                f" [1, {self.kmer_length - 1}]"
            )

        # residual <- prod_o (I - P_{W_o}) beta, one factor at a time
        residual = kmers
        for offset in range(self.kmer_length - length + 1):
            window = range(offset, offset + length)
            outside = tuple(
                axis for axis in range(self.kmer_length) if axis not in window
            )
            residual = residual - residual.mean(dim=outside, keepdim=True)
        return kmers - residual

    def get_pwm(
        self, top: int | None = None, core: int | None = None
    ) -> Tensor:
        r"""The position frequency matrix of the k-mers the model binds.

        Weights each k-mer by its relative affinity and marginalizes to each
        position, so the rows are distributions. `top` restricts the weighting
        to the highest-scoring k-mers, which is the conventional motif logo;
        over the whole table the :math:`|A|^k` background swamps it.

        Above the footprint of the site, it can sit at any of
        :math:`k-\text{core}+1` offsets in the window and
        :class:`~pyprobound.Mode` does not care which, so a plain marginal
        mixes registers. Passing `core` aligns the k-mers over offsets and
        strands first, seeded on the best-scoring one then refined against the
        growing matrix so that no single k-mer anchors the result.

        Args:
            top: Weight only the `top` highest-scoring k-mers.
            core: The footprint of the site, bounding the offsets searched.
                Defaults to `kmer_length`, i.e. strand alignment only.

        Returns:
            A tensor of shape :math:`(\text{width},\text{in_channels})` whose
            rows sum to one, where width is `kmer_length` unless `core` is
            smaller.
        """
        n_kmers = self.in_channels**self.kmer_length
        if top is None:
            top = n_kmers
        if not 1 <= top <= n_kmers:
            raise ValueError(f"top={top} out of range")
        if core is None:
            core = self.kmer_length
        if not 1 <= core <= self.kmer_length:
            raise ValueError(f"core={core} out of range")

        letters = self.alphabet.alphabet
        if self.alphabet.complement:
            mirror = {
                letter: letters[-1 - index]
                for index, letter in enumerate(letters)
            }
        else:
            mirror = {letter: letter for letter in letters}

        def reverse_complement(word: str) -> str:
            return "".join(mirror[letter] for letter in reversed(word))

        scores = self.get_kmer_scores()
        words = list(scores)[:top]
        weights = torch.softmax(
            torch.tensor([scores[word] for word in words]), dim=0
        )
        encoded = {}
        for word in words:
            for variant in (word, reverse_complement(word)):
                encoded[variant] = [letters.index(i) for i in variant]

        slack = self.kmer_length - core
        width = self.kmer_length + 2 * slack

        def accumulate(placed: dict[str, tuple[int, str]]) -> Tensor:
            counts = torch.full((width, len(letters)), _PSEUDO)
            for weight, word in zip(weights, words):
                offset, variant = placed[word]
                for index, letter in enumerate(encoded[variant]):
                    counts[offset + index, letter] += weight
            return counts

        # seed the alignment on the best-scoring k-mer
        reference = words[0]
        placed: dict[str, tuple[int, str]] = {}
        for word in words:
            best: tuple[float, int, str] | None = None
            for variant in sorted({word, reverse_complement(word)}):
                for shift in range(-slack, slack + 1):
                    match = float(
                        sum(
                            variant[i] == reference[i + shift]
                            for i in range(self.kmer_length)
                            if 0 <= i + shift < self.kmer_length
                        )
                    )
                    if best is None or match > best[0]:
                        best = (match, slack - shift, variant)
            assert best is not None
            placed[word] = (best[1], best[2])

        # then refine against the matrix, so an outlier cannot anchor it
        for _ in range(_REFINEMENTS):
            counts = accumulate(placed)
            log_pwm = torch.log(counts / counts.sum(dim=1, keepdim=True))
            updated: dict[str, tuple[int, str]] = {}
            for word in words:
                best = None
                for variant in sorted({word, reverse_complement(word)}):
                    indices = encoded[variant]
                    for offset in range(width - self.kmer_length + 1):
                        score = float(
                            sum(
                                log_pwm[offset + index, letter]
                                for index, letter in enumerate(indices)
                            )
                        )
                        if best is None or score > best[0]:
                            best = (score, offset, variant)
                assert best is not None
                updated[word] = (best[1], best[2])
            if updated == placed:
                break
            placed = updated

        counts = accumulate(placed)
        depth = counts.sum(dim=1)
        counts = counts[depth >= _MIN_SUPPORT * depth.max()]
        return counts / counts.sum(dim=1, keepdim=True)

    def to_psam(
        self,
        top: int | None = None,
        core: int | None = None,
        name: str | None = None,
    ) -> PSAM:
        r"""A PSAM summarizing the table, for `pyprobound.plotting.logo`.

        Sets the monomer betas to the log position frequencies of the k-mers
        the model binds, which discards every interaction between positions --
        the whole reason to use a table. Read it beside `get_kmer_scores`. Pass
        `core`, the footprint of the site, to offset-align first.
        """
        matrix = self.get_pwm(top=top, core=core)
        psam = PSAM(
            kernel_size=len(matrix),
            alphabet=self.alphabet,
            score_reverse=False,
            name=self.name if name is None else name,
        )
        betas = torch.log(matrix.clamp(min=1e-9))
        with torch.no_grad():
            for position in range(len(matrix)):
                symmetry = psam.symmetry[position]
                for index in range(self.in_channels):
                    # pylint: disable-next=protected-access
                    key = PSAM._get_key((symmetry, symmetry), 0, index)
                    psam.betas[key].fill_(float(betas[position, index]))
        return psam

    def get_kmer_scores(self) -> "dict[str, float]":
        """The score of every k-mer, keyed by its string, sorted descending."""
        with torch.inference_mode():
            table = self.get_table()[0].flatten().cpu()
        letters = self.alphabet.alphabet
        base = len(letters)
        scores: dict[str, float] = {}
        for index in torch.argsort(table, descending=True).tolist():
            kmer, remainder = "", index
            for _ in range(self.kmer_length):
                kmer = letters[remainder % base] + kmer
                remainder //= base
            scores[kmer] = table[index].item()
        return scores


class KmerScoring(Layer):
    r"""Scores a :class:`Kmers` table over every sliding window of a sequence.

    The output of window :math:`x` of sequence :math:`i` is the
    :math:`-\log K^{rel}_{\text{D}}` of that window,

    .. math::
        \log \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}
        = \beta_{S_{i,x}}

    where :math:`\beta` is the k-mer table.
    """

    def __init__(
        self,
        layer_spec: Kmers,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
    ) -> None:
        """Initializes the k-mer layer.

        Args:
            layer_spec: The specification of the k-mer layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
        """
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )
        self.layer_spec: Kmers

    @classmethod
    def from_spec(cls, spec: Kmers, prev: Table[Any] | Layer) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the k-mer layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
        """
        if isinstance(prev, Layer):
            input_shape = prev.out_len(prev.input_shape, "shape")
            min_input_length = prev.out_len(prev.min_input_length, "min")
            max_input_length = prev.out_len(prev.max_input_length, "max")
        else:
            input_shape = prev.input_shape
            min_input_length = prev.min_read_length
            max_input_length = prev.max_read_length
        if isinstance(prev, Table):
            if (
                spec.init_from_counts
                and not spec.counts_initialized
                and isinstance(prev, CountBatch)
            ):
                spec.init_from_counts_(prev)
            n_kmers = spec.in_channels**spec.kmer_length
            windows = max(spec.out_len(input_shape), 0)
            per_kmer = len(prev) * windows / n_kmers
            if per_kmer < _MIN_WINDOWS_PER_KMER:
                warnings.warn(
                    f"kmer_length={spec.kmer_length} gives {n_kmers} k-mers"
                    f" but only {per_kmer:.1f} windows per k-mer in a table of"
                    f" {len(prev)} sequences, so the table is unlikely to be"
                    " estimable. Reduce kmer_length, or reduce the loss's"
                    " pseudocount, which has to fall as kmer_length rises.",
                    stacklevel=2,
                )
        return cls(
            layer_spec=spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        binding_optim = self.layer_spec.update_binding_optim(binding_optim)
        binding_optim.merge_binding_optim()
        return binding_optim

    def score_dense(self, seqs: Tensor) -> Tensor:
        r"""Scores integer-encoded sequences by indexing into the table.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """
        kmer_length = self.layer_spec.kmer_length
        if cast(
            bool, (seqs >= self.layer_spec.in_channels).any().item()
        ):  # ' ', '*', or '-' present
            table = self.layer_spec.get_padded_table()
        else:
            table = self.layer_spec.get_table()

        index = seqs.unfold(-1, kmer_length, 1).unbind(-1)
        # One gather per output channel; out_channels is 1 or 2
        return torch.stack(
            [channel[index] for channel in table.unbind(0)], dim=1
        )

    def score_onehot(self, seqs: Tensor) -> Tensor:
        r"""Scores probability-encoded sequences by contracting the table.

        The expected score of each window under the per-position
        distributions. Materializes
        :math:`\text{minibatch}\times\text{out_length}\times|A|^{k-1}`, so it
        suits the small batches used for :math:`\mathbb{E}[\text{score}]`.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """
        kmer_length = self.layer_spec.kmer_length
        table = self.layer_spec.get_table()
        n_alpha = self.layer_spec.in_channels

        finite = seqs.isfinite().all(dim=1)  # (minibatch, length)
        probs = seqs.masked_fill(~seqs.isfinite(), 0.0)
        unfold = probs.unfold(2, kmer_length, 1)  # (minibatch, |A|, x, k)
        minibatch, _, out_length, _ = unfold.shape

        # Fast path for a uniform prior over channels, as returned by
        # Mode.expected_sequence(); avoids the |A|^k contraction below.
        if torch.all(probs == 1 / n_alpha):
            result = table.mean(
                dim=tuple(range(1, kmer_length + 1))
            )  # (out_channels,)
            result = result.reshape(1, -1, 1).expand(minibatch, -1, out_length)
        else:
            # Fold one position at a time; `acc` is
            # (out_channels, |A|, |A|^(k-1-j), minibatch, out_length)
            acc = torch.tensordot(
                table, unfold[..., 0], dims=([1], [1])
            )  # (out_channels, |A|^(k-1), minibatch, x)
            for position in range(1, kmer_length):
                weight = unfold[..., position].permute(1, 0, 2)
                acc = acc.reshape(
                    table.shape[0], n_alpha, -1, minibatch, out_length
                )
                acc = (
                    acc * weight.reshape(1, n_alpha, 1, minibatch, out_length)
                ).sum(dim=1)
            result = acc.reshape(
                table.shape[0], minibatch, out_length
            ).permute(1, 0, 2)

        window_finite = finite.unfold(-1, kmer_length, 1).all(dim=-1)
        return result.masked_fill(~window_finite.unsqueeze(1), float("-inf"))

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """
        if seqs.ndim == 3:
            return self.score_onehot(seqs)
        return self.score_dense(seqs)
