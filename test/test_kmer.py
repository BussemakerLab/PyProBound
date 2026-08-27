# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, missing-module-docstring
import unittest
import warnings
from typing import Any, cast

import torch
from typing_extensions import override

import pyprobound.layers
from pyprobound.layers.kmer import _reverse_complement

from . import make_count_table
from .test_layers import BaseTestCases


def make_layer(
    kmer_length: int = 4, **kwargs: object
) -> tuple[pyprobound.CountTable, pyprobound.layers.KmerScoring]:
    count_table = make_count_table()
    spec = pyprobound.layers.Kmers(
        alphabet=count_table.alphabet,
        kmer_length=kmer_length,
        **kwargs,  # type: ignore[arg-type]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*windows per k-mer.*")
        return count_table, pyprobound.layers.KmerScoring.from_spec(
            spec, count_table
        )


class TestKmerScoring(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer()


class TestKmerScoring_rcsym(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(rc_symmetry=True)


class TestKmerScoring_legacy(BaseTestCases.BaseTestLayer):
    """No reverse strand and no mean-centering, as in the original layer."""

    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer(normalize=False)


class TestKmerScoring_onehot(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table, self.layer = make_layer()
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)


class TestKmers(unittest.TestCase):
    def test_defaults(self) -> None:
        _, layer = make_layer()
        self.assertTrue(
            layer.layer_spec.rc_symmetry,
            "a complementable alphabet should default to rc_symmetry",
        )
        self.assertEqual(layer.out_channels, 1)
        self.assertEqual(layer.layer_spec.backoff_length, 3)
        with self.assertRaises(ValueError):
            make_layer(kmer_length=4, backoff_length=4)

    def test_score_dense(self) -> None:
        """Both output channels match an explicit per-window lookup."""
        count_table, layer = make_layer(kmer_length=4, rc_symmetry=False)
        table = layer.layer_spec.get_table()
        seqs = torch.randint(0, 4, (8, count_table.input_shape))
        out = layer(seqs)
        self.assertEqual(out.shape[1], 1)
        for i in range(len(seqs)):
            for window in range(layer.out_len(seqs.shape[1])):
                kmer = seqs[i, window : window + 4].tolist()
                self.assertAlmostEqual(
                    out[i, 0, window].item(),
                    table[0][tuple(kmer)].item(),
                    places=5,
                    msg="incorrect window score",
                )

    def test_score_onehot_matches_dense(self) -> None:
        count_table, layer = make_layer()
        seqs = torch.randint(0, 4, (8, count_table.input_shape))
        onehot = (
            torch.nn.functional.one_hot(seqs, num_classes=4)
            .transpose(1, 2)
            .to(layer.layer_spec.kmers.dtype)
        )
        self.assertTrue(
            torch.allclose(layer(seqs), layer(onehot), atol=1e-5),
            "one-hot scoring disagrees with dense scoring",
        )

    def test_score_onehot_uniform(self) -> None:
        """The uniform fast path matches the general contraction."""
        count_table, layer = make_layer(normalize=False)
        uniform = torch.full((1, 4, count_table.input_shape), 0.25)
        nudged = uniform.clone()
        nudged[0, 0, 0] += 1e-9
        self.assertTrue(
            torch.allclose(layer(uniform), layer(nudged), atol=1e-6),
            "uniform fast path disagrees with the general contraction",
        )

    def test_padding(self) -> None:
        """' ' is not scored; '*' and '-' average over their position."""
        _, layer = make_layer(kmer_length=4)
        table = layer.layer_spec.get_table()[0]
        alphabet = layer.layer_spec.alphabet
        seqs = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3]
                + [alphabet.neginf_pad] * (layer.input_shape - 8)
            ]
        )
        out = layer(seqs)[0, 0]
        self.assertAlmostEqual(
            out[0].item(), table[0, 1, 2, 3].item(), places=5
        )
        self.assertTrue(
            torch.isinf(out[-1]) and out[-1] < 0,
            "a window inside ' ' padding should not be scored",
        )

        for pad in (alphabet.wildcard_pad, alphabet.zero_pad):
            seqs = torch.tensor(
                [[pad, pad, 0, 1] + [2] * (layer.input_shape - 4)]
            )
            self.assertAlmostEqual(
                layer(seqs)[0, 0, 0].item(),
                table[:, :, 0, 1].mean().item(),
                places=5,
                msg=f"padding index {pad} should average over its position",
            )

    def test_rc_symmetry(self) -> None:
        _, layer = make_layer(rc_symmetry=True)
        table = layer.layer_spec.get_table()[0]
        self.assertTrue(
            torch.allclose(
                table,
                _reverse_complement(table, layer.layer_spec.kmer_length),
                atol=1e-6,
            ),
            "table is not reverse-complement symmetric",
        )

    def test_normalize(self) -> None:
        _, layer = make_layer()
        self.assertAlmostEqual(
            layer.layer_spec.get_table()[0].mean().item(),
            0.0,
            places=6,
            msg="normalize=True should mean-center the table",
        )

    def test_get_pwm(self) -> None:
        """Rows are distributions, and the motif dominates the bound set."""
        count_table = make_count_table()
        spec = pyprobound.layers.Kmers(
            alphabet=count_table.alphabet,
            kmer_length=6,
            init_from_counts=False,
            rc_symmetry=False,
            normalize=False,
        )
        with torch.no_grad():
            spec.kmers.zero_()
            index = tuple("ACGT".index(letter) for letter in "CACGTG")
            spec.kmers[index] = 8.0

        pwm = spec.get_pwm()
        self.assertEqual(tuple(pwm.shape), (6, 4))
        self.assertTrue(
            torch.allclose(pwm.sum(dim=1), torch.ones(6), atol=1e-5),
            "rows should be distributions",
        )
        self.assertEqual(
            "".join("ACGT"[int(row.argmax())] for row in pwm),
            "CACGTG",
            "the consensus of the bound population should be the seeded motif",
        )

        # restricting to the top k-mers sharpens it; temperature flattens it
        def information(matrix: torch.Tensor) -> float:
            clamped = matrix.clamp(min=1e-12)
            return float((2.0 + (clamped * clamped.log2()).sum(dim=1)).sum())

        self.assertGreater(
            information(spec.get_pwm(top=10)),
            information(spec.get_pwm()),
            "top should concentrate the weighting",
        )
        with self.assertRaises(ValueError):
            spec.get_pwm(top=0)

    def test_to_psam(self) -> None:
        """A PSAM summary that pyprobound.plotting.logo can draw."""
        count_table = make_count_table()
        spec = pyprobound.layers.Kmers(
            alphabet=count_table.alphabet,
            kmer_length=6,
            init_from_counts=False,
            rc_symmetry=False,
            normalize=False,
            name="test",
        )
        with torch.no_grad():
            spec.kmers.zero_()
            spec.kmers[tuple("ACGT".index(i) for i in "CACGTG")] = 8.0

        psam = spec.to_psam(top=50)
        self.assertEqual(psam.kernel_size, 6)
        self.assertEqual(psam.pairwise_distance, 0)
        self.assertEqual(psam.out_channels // psam.n_strands, 1)

        # the filter must carry the same consensus as the PWM
        matrix = psam.get_filter(0)[0]
        self.assertEqual(
            "".join("ACGT"[int(column.argmax())] for column in matrix.T),
            "CACGTG",
            "to_psam lost the consensus",
        )
        # and the same shape, up to the per-position constant a logo removes
        expected = torch.log(spec.get_pwm(top=50).clamp(min=1e-9))
        self.assertTrue(
            torch.allclose(
                matrix.T - matrix.T.mean(dim=1, keepdim=True),
                expected - expected.mean(dim=1, keepdim=True),
                atol=1e-4,
            ),
            "to_psam betas do not match log get_pwm",
        )

    def test_get_pwm_aligned(self) -> None:
        """Offset-aligning the top k-mers recovers a single register."""
        count_table = make_count_table()
        spec = pyprobound.layers.Kmers(
            alphabet=count_table.alphabet,
            kmer_length=8,
            init_from_counts=False,
            rc_symmetry=False,
            normalize=False,
        )
        # the same 6-mer core at two different offsets, as the sliding-window
        # logsumexp allows, the stronger one first
        with torch.no_grad():
            spec.kmers.zero_()
            core = ["ACGT".index(letter) for letter in "CACGTG"]
            for offset, value in ((0, 8.0), (2, 7.5)):
                index: list[Any] = [slice(None)] * 8
                for position, letter in enumerate(core):
                    index[offset + position] = letter
                spec.kmers[tuple(index)] = value

        matrix = spec.get_pwm(top=32, core=6)
        self.assertTrue(
            torch.allclose(
                matrix.sum(dim=1), torch.ones(len(matrix)), atol=1e-5
            ),
            "rows should be distributions",
        )
        self.assertLessEqual(len(matrix), 3 * 8 - 2 * 6)
        self.assertIn(
            "CACGTG",
            "".join("ACGT"[int(row.argmax())] for row in matrix),
            "aligning should recover the core",
        )
        for bad in (0, 9):
            with self.assertRaises(ValueError):
                spec.get_pwm(core=bad)


class TestKmerRegularization(unittest.TestCase):
    def test_get_dirichlet_penalizes_the_backoff_residual(self) -> None:
        _, layer = make_layer(init_from_counts=False)
        spec = layer.layer_spec
        self.assertLess(
            spec.get_dirichlet().item(),
            0.0,
            "a log-density should be negative",
        )
        # exactly representable by the backoff target, so nothing to penalize
        with torch.no_grad():
            spec.kmers.copy_(spec.get_backoff())
        self.assertAlmostEqual(
            spec.get_dirichlet().item(),
            0.0,
            places=5,
            msg="the backoff target itself should not be penalized",
        )

    def test_backoff_target_is_an_orthogonal_projection(self) -> None:
        """P_{k-1} is idempotent and free for any sum of two (k-1)-mer fns."""
        _, layer = make_layer(
            kmer_length=4, rc_symmetry=False, normalize=False
        )
        spec = layer.layer_spec
        with torch.no_grad():
            spec.kmers.copy_(torch.randn_like(spec.kmers) * 2)
        target = spec.get_backoff()

        _, again = make_layer(
            kmer_length=4, rc_symmetry=False, normalize=False
        )
        with torch.no_grad():
            again.layer_spec.kmers.copy_(target)
        self.assertTrue(
            torch.allclose(again.layer_spec.get_backoff(), target, atol=1e-5),
            "P_{k-1} is not idempotent",
        )

        # f(x1..x3) + g(x2..x4) is representable, so it is free
        first = torch.randn(4, 4, 4)
        second = torch.randn(4, 4, 4)
        with torch.no_grad():
            spec.kmers.copy_(first.unsqueeze(-1) + second.unsqueeze(0))
        self.assertAlmostEqual(
            spec.get_dirichlet().item(),
            0.0,
            places=5,
            msg="a sum of two (k-1)-mer functions should not be penalized",
        )
        # a pure first-to-last coupling is penalized
        with torch.no_grad():
            spec.kmers[0, :, :, 0] += 3.0
        self.assertLess(
            spec.get_dirichlet().item(),
            -1e-6,
            "a first-to-last interaction should be penalized",
        )

    def test_backoff_length_nests(self) -> None:
        """Lower backoff_length shrinks harder; each is a projection."""
        _, layer = make_layer(
            kmer_length=5, rc_symmetry=False, normalize=False
        )
        spec = layer.layer_spec
        with torch.no_grad():
            spec.kmers.copy_(torch.randn_like(spec.kmers) * 2)
        residuals = []
        for length in range(1, 5):
            target = spec.get_backoff(length)
            residual = spec.kmers.detach() - target
            residuals.append(residual.std().item())
            self.assertAlmostEqual(
                (residual * target).sum().item(),
                0.0,
                places=3,
                msg=f"residual not orthogonal to target at m={length}",
            )
            # a function of one contiguous window of this length is free
            other = make_layer(
                kmer_length=5,
                rc_symmetry=False,
                normalize=False,
                backoff_length=length,
            )[1].layer_spec
            block = torch.randn(*([4] * length))
            shape = [1] * 5
            for j in range(length):
                shape[j] = 4
            with torch.no_grad():
                other.kmers.copy_(
                    block.reshape(shape).expand((4,) * 5).contiguous()
                )
            self.assertAlmostEqual(
                other.get_dirichlet().item(),
                0.0,
                places=4,
                msg=f"a {length}-window function should be free at m={length}",
            )
        self.assertEqual(
            residuals,
            sorted(residuals, reverse=True),
            "residual should shrink as backoff_length rises",
        )

    def test_backoff_length_validation(self) -> None:
        with self.assertRaises(ValueError):
            make_layer(kmer_length=4, backoff_length=0)
        with self.assertRaises(ValueError):
            make_layer(kmer_length=4, backoff_length=4)
        _, layer = make_layer(kmer_length=4)
        self.assertEqual(layer.layer_spec.backoff_length, 3)

    def test_dirichlet_gradient(self) -> None:
        for length in (1, 2):
            self._check_dirichlet_gradient(length)

    def _check_dirichlet_gradient(self, backoff_length: int) -> None:
        _, layer = make_layer(kmer_length=3, backoff_length=backoff_length)
        spec = layer.layer_spec
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            spec.kmers.data = spec.kmers.data.double()
            spec.unfreeze("kmers")
            spec.kmers.grad = None
            spec.get_dirichlet().backward()  # type: ignore[no-untyped-call]
            self.assertIsNotNone(spec.kmers.grad)
            analytical = cast(torch.Tensor, spec.kmers.grad)
            flat = spec.kmers.data.view(-1)
            for index in range(flat.numel()):
                original = flat[index].item()
                flat[index] = original + 1e-6
                high = spec.get_dirichlet().item()
                flat[index] = original - 1e-6
                low = spec.get_dirichlet().item()
                flat[index] = original
                self.assertAlmostEqual(
                    (high - low) / 2e-6,
                    analytical.view(-1)[index].item(),
                    places=6,
                    msg=f"m={backoff_length}: autograd disagrees with"
                    " finite differences",
                )

    def test_dirichlet_ignores_unidentifiable_component(self) -> None:
        """Regularization must act on the symmetrized table."""
        _, layer = make_layer(rc_symmetry=True)
        spec = layer.layer_spec
        before = spec.get_dirichlet().item()
        antisymmetric = torch.randn_like(spec.kmers)
        antisymmetric = antisymmetric - _reverse_complement(
            antisymmetric, spec.kmer_length
        )
        with torch.no_grad():
            spec.kmers.add_(antisymmetric)
        self.assertAlmostEqual(
            spec.get_dirichlet().item(),
            before,
            places=4,
            msg="regularization moved with the unidentifiable component",
        )

    def test_unidentifiable_component_gets_no_gradient(self) -> None:
        """Under RC symmetry the gradient is itself RC-symmetric."""
        count_table, layer = make_layer(rc_symmetry=True)
        spec = layer.layer_spec
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            spec.unfreeze("kmers")
            out = layer(torch.randint(0, 4, (8, count_table.input_shape)))
            out.sum().backward()  # type: ignore[no-untyped-call]
        self.assertIsNotNone(spec.kmers.grad)
        grad = cast(torch.Tensor, spec.kmers.grad)
        self.assertTrue(
            torch.allclose(
                grad, _reverse_complement(grad, spec.kmer_length), atol=1e-6
            ),
            "gradient has a component along an unidentifiable direction",
        )


class TestKmerInitialization(unittest.TestCase):
    def test_init_from_counts(self) -> None:
        """Starts at the observed log enrichment, deterministically."""
        count_table = make_count_table(n_seqs=200)
        specs = [
            pyprobound.layers.Kmers(
                alphabet=count_table.alphabet, kmer_length=2
            )
            for _ in range(2)
        ]
        for spec in specs:
            pyprobound.layers.KmerScoring.from_spec(spec, count_table)
        self.assertTrue(
            specs[0].counts_initialized, "counts init was not applied"
        )
        self.assertTrue(
            torch.equal(specs[0].kmers, specs[1].kmers),
            "counts init is not deterministic",
        )

        # An enriched 2-mer must outscore a depleted one.
        windows = count_table.seqs.unfold(-1, 2, 1)
        valid = (windows < 4).all(dim=-1)
        index = (windows * torch.tensor([4, 1])).sum(dim=-1)[valid]
        rows = (
            torch.arange(len(count_table.seqs))
            .unsqueeze(1)
            .expand_as(valid)[valid]
        )
        first, last = (
            torch.bincount(
                index,
                weights=count_table.target[rows, col].double(),
                minlength=16,
            )
            for col in (0, -1)
        )
        expected = torch.log((last + 1) / (first + 1))
        expected = expected - expected.mean()
        self.assertTrue(
            torch.allclose(
                specs[0].kmers.flatten().double(), expected, atol=1e-5
            ),
            "counts init does not match the observed log enrichment",
        )

    def test_warns_when_table_is_not_estimable(self) -> None:
        count_table = make_count_table(n_seqs=100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyprobound.layers.KmerScoring.from_spec(
                pyprobound.layers.Kmers(
                    alphabet=count_table.alphabet, kmer_length=8
                ),
                count_table,
            )
        self.assertTrue(
            any("windows per k-mer" in str(w.message) for w in caught),
            "no warning for a table that cannot be estimated",
        )

    def test_no_warning_when_table_is_estimable(self) -> None:
        count_table = make_count_table(n_seqs=100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyprobound.layers.KmerScoring.from_spec(
                pyprobound.layers.Kmers(
                    alphabet=count_table.alphabet, kmer_length=2
                ),
                count_table,
            )
        self.assertFalse(
            any("windows per k-mer" in str(w.message) for w in caught),
            "spurious warning for an estimable table",
        )

    def test_gradient(self) -> None:
        """Autograd agrees with finite differences on both score paths."""
        count_table, layer = make_layer(kmer_length=3)
        spec = layer.layer_spec
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            spec.kmers.data = spec.kmers.data.double()
            spec.unfreeze("kmers")
            seqs = torch.randint(0, 4, (4, count_table.input_shape))
            onehot = torch.rand(
                2, 4, count_table.input_shape, dtype=torch.double
            )
            onehot /= onehot.sum(dim=1, keepdim=True)

            for inputs in (seqs, onehot):
                weight = torch.randn(
                    len(inputs),
                    layer.out_channels,
                    layer.out_len(count_table.input_shape),
                    dtype=torch.double,
                )

                def loss(
                    inputs: torch.Tensor = inputs,
                    weight: torch.Tensor = weight,
                ) -> torch.Tensor:
                    return (layer(inputs) * weight).sum()

                spec.kmers.grad = None
                loss().backward()  # type: ignore[no-untyped-call]
                self.assertIsNotNone(spec.kmers.grad)
                analytical = cast(torch.Tensor, spec.kmers.grad)

                flat = spec.kmers.data.view(-1)
                for index in range(flat.numel()):
                    original = flat[index].item()
                    flat[index] = original + 1e-6
                    high = loss().item()
                    flat[index] = original - 1e-6
                    low = loss().item()
                    flat[index] = original
                    self.assertAlmostEqual(
                        (high - low) / 2e-6,
                        analytical.view(-1)[index].item(),
                        places=5,
                        msg="autograd disagrees with finite differences",
                    )


if __name__ == "__main__":
    unittest.main()
