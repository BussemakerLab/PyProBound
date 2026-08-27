# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, missing-module-docstring
import itertools
import unittest
from typing import Any

import torch
from typing_extensions import override

import pyprobound.layers

from . import make_count_table
from .test_layers import BaseTestCases


def make_layer(
    kernel_size: int = 4,
    layer_kwargs: dict[str, object] | None = None,
    fixed_length: bool = False,
    **kwargs: object,
) -> tuple[pyprobound.CountTable, pyprobound.layers.IsingConv1d]:
    # the analytic references below multiply the filter by a one-hot encoding,
    # so they need input without -inf padding
    count_table = (
        make_count_table(min_input_length=24, max_input_length=24)
        if fixed_length
        else make_count_table()
    )
    spec = pyprobound.layers.IsingPSAM(
        kernel_size=kernel_size,
        alphabet=count_table.alphabet,
        **kwargs,  # type: ignore[arg-type]
    )
    return count_table, pyprobound.layers.IsingConv1d.from_psam(
        spec, count_table, **(layer_kwargs or {})  # type: ignore[arg-type]
    )


def reference_log_score(
    spec: pyprobound.layers.IsingPSAM, betas: torch.Tensor, reverse: bool
) -> torch.Tensor:
    """Enumerates every microstate explicitly, independent of the layer."""
    coupling = spec.interface_coupling()
    if reverse:
        coupling = coupling.flip(0)
    total = []
    for sigma in itertools.product([1.0, -1.0], repeat=betas.shape[-1]):
        spins = torch.tensor(sigma)
        on = ((spins + 1) / 2 * betas).sum(-1)
        pair = (
            ((spins[:-1] * spins[1:] - 1) * coupling).sum(-1)
            if len(spins) > 1
            else torch.zeros_like(on)
        )
        total.append(on + pair)
    return torch.logsumexp(torch.stack(total, dim=-1), dim=-1)


class TestIsingConv1d(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(coupling=1.3)


class TestIsingConv1d_enumerate(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(
            coupling=1.3, method="enumerate"
        )


class TestIsingConv1d_vector(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(
            coupling=torch.tensor([0.2, -1.1, 2.7])
        )


class TestIsingConv1d_onehot(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(
            coupling=1.3, layer_kwargs={"one_hot": True}
        )
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)


class TestIsingPSAM(unittest.TestCase):
    def test_defaults(self) -> None:
        _, layer = make_layer()
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        self.assertEqual(spec.method, "transfer")
        self.assertEqual(spec.coupling.numel(), 1)
        self.assertEqual(spec.n_expanded, spec.kernel_size)
        torch.testing.assert_close(
            spec.interface_coupling(), torch.zeros(spec.kernel_size - 1)
        )

    def test_coupling_groups_infers_from_length(self) -> None:
        _, layer = make_layer(kernel_size=5, coupling=torch.arange(4.0))
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        self.assertEqual(spec.coupling.numel(), 4)
        torch.testing.assert_close(
            spec.interface_coupling(), torch.arange(4.0)
        )

    def test_coupling_groups_shared(self) -> None:
        _, layer = make_layer(
            kernel_size=5,
            coupling=torch.tensor([1.0, 3.0]),
            coupling_groups=torch.tensor([0, 0, 1, 1]),
        )
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        torch.testing.assert_close(
            spec.interface_coupling(), torch.tensor([1.0, 1.0, 3.0, 3.0])
        )

    def test_coupling_length_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "coupling has 2 elements"):
            make_layer(kernel_size=5, coupling=torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "coupling_groups must have"):
            make_layer(kernel_size=5, coupling_groups=torch.zeros(9))

    def test_rejects_pairwise(self) -> None:
        with self.assertRaisesRegex(ValueError, "pairwise"):
            make_layer(pairwise_distance=1)

    def test_rejects_footprint_heuristics(self) -> None:
        with self.assertRaisesRegex(ValueError, "increment_footprint"):
            make_layer(increment_footprint=True)

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown method"):
            make_layer(method="matrix")

    def test_get_logo_filter_is_unexpanded(self) -> None:
        _, layer = make_layer(kernel_size=6)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        self.assertEqual(
            spec.get_logo_filter().shape,
            (spec.n_strands, spec.in_channels, spec.kernel_size),
        )
        self.assertEqual(
            spec.get_filter(0).shape[0], spec.out_channels * spec.n_expanded
        )

    def test_set_coupling(self) -> None:
        _, layer = make_layer(kernel_size=5, coupling=torch.arange(4.0))
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        spec.set_coupling(2.5)
        torch.testing.assert_close(
            spec.interface_coupling(), torch.full((4,), 2.5)
        )


class TestLogoFilter(unittest.TestCase):
    """`get_logo_filter` must not change what a standard PSAM plots."""

    def test_matches_get_filter_on_a_standard_psam(self) -> None:
        count_table = make_count_table()
        cases: dict[str, dict[str, Any]] = {
            "plain": {"kernel_size": 6},
            "pairwise": {"kernel_size": 6, "pairwise_distance": 2},
            "symmetry": {"kernel_size": 6, "symmetry": [1, 2, 3, -3, -2, -1]},
            "no_reverse": {"kernel_size": 6, "score_reverse": False},
            "multichannel": {"kernel_size": 6, "out_channels": 4},
            "not_normalized": {"kernel_size": 6, "normalize": False},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                psam = pyprobound.layers.PSAM(
                    alphabet=count_table.alphabet, **kwargs
                )
                for beta in psam.betas.values():
                    beta.copy_(torch.randn(()))
                for dist in range(psam.pairwise_distance + 1):
                    torch.testing.assert_close(
                        psam.get_logo_filter(dist), psam.get_filter(dist)
                    )

    def test_ising_logo_filter_is_not_the_expanded_filter(self) -> None:
        _, layer = make_layer(kernel_size=6)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        self.assertNotEqual(
            spec.get_logo_filter().shape[0], spec.get_filter(0).shape[0]
        )


class TestIsingScoring(unittest.TestCase):
    """Both reductions against an independent enumeration of every state."""

    def _check_against_reference(self, **kwargs: Any) -> None:
        count_table, layer = make_layer(fixed_length=True, **kwargs)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        for beta in spec.betas.values():
            beta.copy_(torch.randn(()) * 2)

        out = layer(count_table.seqs)

        # score each position of each window with the expanded filter
        expanded = spec.get_filter(0)
        per_position = torch.nn.functional.conv1d(
            count_table.alphabet.embedding(count_table.seqs).transpose(1, 2),
            expanded,
        ).reshape(len(count_table), spec.out_channels, spec.n_expanded, -1)
        self.assertEqual(spec.method, "transfer")
        betas = per_position.movedim(2, -1)
        for channel in range(spec.out_channels):
            expected = reference_log_score(
                spec, betas[:, channel], spec.is_reverse(channel)
            )
            torch.testing.assert_close(
                out[:, channel], expected, atol=1e-4, rtol=1e-4
            )

    def test_transfer_matches_enumeration(self) -> None:
        for kernel_size in (2, 4, 8):
            with self.subTest(kernel_size=kernel_size):
                self._check_against_reference(
                    kernel_size=kernel_size, coupling=0.7
                )

    def test_transfer_matches_enumeration_vector(self) -> None:
        self._check_against_reference(
            kernel_size=5, coupling=torch.tensor([1.0, -0.5, 2.0, 0.3])
        )

    def test_methods_agree(self) -> None:
        for coupling in (0.0, 1.4, torch.tensor([0.5, -1.0, 2.2])):
            with self.subTest(coupling=coupling):
                count_table, transfer = make_layer(
                    coupling=coupling, method="transfer"
                )
                _, enumerate_ = make_layer(
                    coupling=coupling, method="enumerate"
                )
                spec_t = transfer.layer_spec
                spec_e = enumerate_.layer_spec
                assert isinstance(spec_t, pyprobound.layers.IsingPSAM)
                assert isinstance(spec_e, pyprobound.layers.IsingPSAM)
                for key, beta in spec_t.betas.items():
                    beta.copy_(torch.randn(()) * 2)
                    spec_e.betas[key].copy_(beta)
                torch.testing.assert_close(
                    transfer(count_table.seqs),
                    enumerate_(count_table.seqs),
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_onehot_matches_dense(self) -> None:
        count_table, layer = make_layer(coupling=1.1)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        for beta in spec.betas.values():
            beta.copy_(torch.randn(()) * 2)
        dense = layer(count_table.seqs)
        layer.one_hot = True
        onehot = layer(
            count_table.alphabet.embedding(count_table.seqs).transpose(1, 2)
        )
        layer.one_hot = False
        torch.testing.assert_close(dense, onehot)

    def test_zero_coupling_is_independent_positions(self) -> None:
        """At :math:`J = 0` each position contributes softplus(beta)."""
        count_table, layer = make_layer(
            kernel_size=6, coupling=0.0, fixed_length=True
        )
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        for beta in spec.betas.values():
            beta.copy_(torch.randn(()) * 2)
        per_position = torch.nn.functional.conv1d(
            count_table.alphabet.embedding(count_table.seqs).transpose(1, 2),
            spec.get_filter(0),
        ).reshape(len(count_table), spec.out_channels, spec.n_expanded, -1)
        expected = torch.nn.functional.softplus(per_position).sum(2)
        torch.testing.assert_close(
            layer(count_table.seqs), expected, atol=1e-4, rtol=1e-4
        )

    def test_large_coupling_is_all_or_none(self) -> None:
        r"""At large :math:`J` the window flips as a whole."""
        count_table, layer = make_layer(
            kernel_size=5, coupling=30.0, fixed_length=True
        )
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        for beta in spec.betas.values():
            beta.copy_(torch.randn(()))
        per_position = torch.nn.functional.conv1d(
            count_table.alphabet.embedding(count_table.seqs).transpose(1, 2),
            spec.get_filter(0),
        ).reshape(len(count_table), spec.out_channels, spec.n_expanded, -1)
        total = per_position.sum(2)
        expected = torch.logaddexp(total, torch.zeros_like(total))
        torch.testing.assert_close(
            layer(count_table.seqs), expected, atol=1e-3, rtol=1e-3
        )

    def test_padding_is_neginf(self) -> None:
        """A window entirely over padding must score -inf, not 0."""
        count_table, layer = make_layer(kernel_size=4, coupling=1.0)
        seqs = torch.nn.functional.pad(
            count_table.seqs, (0, 6), value=count_table.alphabet.neginf_pad
        )
        out = layer(seqs)
        self.assertTrue(out[..., -1].isneginf().all())
        self.assertTrue(out[..., 0].isfinite().all())

    def test_out_channel_indexing(self) -> None:
        count_table, layer = make_layer(coupling=0.9)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        for beta in spec.betas.values():
            beta.copy_(torch.randn(()) * 2)
        full = layer(count_table.seqs)
        indexed = pyprobound.layers.IsingConv1d.from_psam(
            spec, count_table, out_channel_indexing=[1]
        )
        self.assertEqual(indexed.out_channels, 1)
        torch.testing.assert_close(
            indexed(count_table.seqs), full[:, 1:2], atol=1e-5, rtol=1e-5
        )

    def test_multiple_motifs(self) -> None:
        count_table, layer = make_layer(coupling=1.0, out_channels=4)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        self.assertEqual(layer.out_channels, 4)
        self.assertEqual(layer(count_table.seqs).shape[1], spec.out_channels)

    def test_score_reverse_false(self) -> None:
        count_table, layer = make_layer(coupling=1.0, score_reverse=False)
        self.assertEqual(layer.out_channels, 1)
        self.assertTrue(layer(count_table.seqs).isfinite().any())

    def test_posbias_is_added_after_reduction(self) -> None:
        count_table, layer = make_layer(
            coupling=1.2, layer_kwargs={"train_posbias": True}
        )
        layer.log_posbias.copy_(torch.randn_like(layer.log_posbias))
        without = layer(count_table.seqs)
        layer.log_posbias.zero_()
        baseline = layer(count_table.seqs)
        self.assertFalse(torch.allclose(without, baseline))


class TestIsingOptim(unittest.TestCase):
    def test_schedule_appears_in_binding_optim(self) -> None:
        _, layer = make_layer(coupling_schedule=(0.0, 1.0, 2.0))
        mode = pyprobound.Mode([layer])
        optim = next(iter(mode.optim_procedure().values()))
        calls = [
            (call.fun, call.kwargs)
            for step in optim.steps
            for call in step.calls
        ]
        self.assertIn(("set_coupling", {"value": 1.0}), calls)
        self.assertIn(("set_coupling", {"value": 2.0}), calls)
        self.assertIn(("unfreeze", {"parameter": "coupling"}), calls)

    def test_coupling_with_monomer_shares_a_step(self) -> None:
        _, layer = make_layer(coupling_schedule=(), coupling_with_monomer=True)
        mode = pyprobound.Mode([layer])
        optim = next(iter(mode.optim_procedure().values()))
        shared = [
            step
            for step in optim.steps
            if any(
                call.kwargs.get("parameter") == "monomer"
                for call in step.calls
            )
        ]
        self.assertTrue(shared)
        self.assertTrue(
            any(
                call.kwargs.get("parameter") == "coupling"
                for call in shared[0].calls
            )
        )

    def test_train_coupling_false(self) -> None:
        _, layer = make_layer(train_coupling=False, coupling=1.0)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        spec.unfreeze("all")
        self.assertFalse(spec.coupling.requires_grad)

    def test_gradient(self) -> None:
        count_table, layer = make_layer(coupling=1.0)
        spec = layer.layer_spec
        assert isinstance(spec, pyprobound.layers.IsingPSAM)
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            spec.unfreeze("all")
            loss = layer(count_table.seqs).logsumexp(dim=(1, 2)).sum()
            loss.backward()  # type: ignore[no-untyped-call]
        assert spec.coupling.grad is not None
        self.assertTrue(spec.coupling.grad.isfinite().all())
        self.assertNotEqual(float(spec.coupling.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
