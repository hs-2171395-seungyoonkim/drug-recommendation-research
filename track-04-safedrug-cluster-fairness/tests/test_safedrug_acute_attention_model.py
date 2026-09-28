import math
import sys
import types
from pathlib import Path

import dill
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_acute_attention_model import DIAG, SafeDrugAcuteAttention
from models import SafeDrugModel  # SOTA/SafeDrug/src -- on sys.path after the import above


def _tiny_ctor_args(n_diag=6, n_proc=5, n_med=4):
    vocab_size = (n_diag, n_proc, n_med)
    ddi_adj = np.zeros((n_med, n_med))
    ddi_mask_H = np.ones((n_med, 3))
    fingerprints = torch.LongTensor([0, 1])
    adjacency = torch.FloatTensor([[0.0, 1.0], [1.0, 0.0]])
    MPNNSet = [(fingerprints, adjacency, 2) for _ in range(n_med)]
    N_fingerprint = 2
    average_projection = torch.eye(n_med, dtype=torch.float32)
    return vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection


def build_tiny_model(variant="S0", emb_dim=8, **kwargs):
    args = _tiny_ctor_args()
    model = SafeDrugAcuteAttention(*args, emb_dim=emb_dim, device=torch.device("cpu"), variant=variant, **kwargs)
    model.eval()
    return model


def build_tiny_original_model(emb_dim=8):
    args = _tiny_ctor_args()
    model = SafeDrugModel(*args, emb_dim=emb_dim, device=torch.device("cpu"))
    model.eval()
    return model


SAMPLE_INPUT_3 = [
    [[0, 1], [0]],
    [[1, 2, 3], [0, 1]],
]


def _with_hadm(input_3, hadm_ids):
    return [list(adm) + [hadm] for adm, hadm in zip(input_3, hadm_ids)]


def test_s0_forward_matches_original_safedrugmodel_bitwise():
    original = build_tiny_original_model(emb_dim=8)
    variant_model = build_tiny_model(variant="S0", emb_dim=8)
    variant_model.load_state_dict(original.state_dict())
    # self.MPNN_emb is a plain tensor attribute (not a Parameter/buffer),
    # computed once at construction time from a throwaway, randomly
    # initialized MolecularGraphNeuralNetwork -- load_state_dict cannot sync
    # it, so it is copied directly (both fixtures use identical ddi/MPNN
    # inputs, so this is the only difference construction-order randomness
    # could cause).
    variant_model.MPNN_emb = original.MPNN_emb.clone()

    with torch.no_grad():
        out_a, ddi_a = original(SAMPLE_INPUT_3)
        out_b, ddi_b = variant_model(SAMPLE_INPUT_3)

    assert torch.allclose(out_a, out_b, atol=1e-7)
    assert torch.allclose(ddi_a, ddi_b, atol=1e-7)


def test_s0_forward_matches_original_safedrugmodel_in_train_mode_with_matched_dropout():
    # The bitwise-equivalence test above runs both models in .eval() mode,
    # where nn.Dropout is an exact identity -- it is mathematically
    # incapable of catching a dropout-placement bug (e.g. dropout applied
    # after the sum instead of before, or to only one modality). Running
    # both in .train() mode (self.dropout has p=0.5 in SafeDrugModel) with
    # an identical torch.manual_seed reset immediately before each forward
    # call makes their dropout masks match exactly, so equal outputs here
    # actually exercise dropout order/placement, not just the no-dropout
    # eval-mode path.
    original = build_tiny_original_model(emb_dim=8)
    variant_model = build_tiny_model(variant="S0", emb_dim=8)
    variant_model.load_state_dict(original.state_dict())
    variant_model.MPNN_emb = original.MPNN_emb.clone()

    original.train()
    variant_model.train()

    torch.manual_seed(12345)
    out_a, ddi_a = original(SAMPLE_INPUT_3)

    torch.manual_seed(12345)
    out_b, ddi_b = variant_model(SAMPLE_INPUT_3)

    assert torch.allclose(out_a, out_b, atol=1e-7)
    assert torch.allclose(ddi_a, ddi_b, atol=1e-7)


def test_s0_never_populates_last_attention():
    model = build_tiny_model(variant="S0")
    with torch.no_grad():
        model(SAMPLE_INPUT_3)
    assert model.last_attention == {}


def test_s1_attention_weights_sum_to_one_per_modality():
    model = build_tiny_model(variant="S1", emb_dim=8)
    hadm_ids = [1001, 1002]
    model.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(0).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    sample = _with_hadm(SAMPLE_INPUT_3, hadm_ids)
    with torch.no_grad():
        model(sample)
    assert model.last_attention["diag"].shape == (3,)  # visit 1 has 3 diag codes
    assert model.last_attention["proc"].shape == (2,)
    assert np.isclose(model.last_attention["diag"].sum(), 1.0, atol=1e-5)
    assert np.isclose(model.last_attention["proc"].sum(), 1.0, atol=1e-5)


def test_missing_text_path_runs_without_error():
    model = build_tiny_model(variant="S1", emb_dim=8)
    model.load_text_features(
        hadm_ids=[1001],
        emb=np.zeros((1, 768), dtype=np.float32),
        has_text=np.array([0], dtype=np.int8),  # has_text False -- must use no_text_embedding
    )
    sample = _with_hadm(SAMPLE_INPUT_3, [1001, 9999])  # 9999 not registered at all either
    with torch.no_grad():
        out, _ = model(sample)
    assert out.shape == (1, 4)  # n_med=4


def test_planted_code_dominates_attention():
    model = build_tiny_model(variant="S1", emb_dim=8)
    hadm_ids = [2001]
    text_emb = np.zeros((1, 768), dtype=np.float32)
    model.load_text_features(hadm_ids=hadm_ids, emb=text_emb, has_text=np.array([1], dtype=np.int8))

    with torch.no_grad():
        # Force W_q to produce a large, known query vector regardless of the
        # (zero) text input, by setting its bias directly.
        model.W_q.weight.zero_()
        q_target = torch.zeros(8)
        q_target[0] = 10.0
        model.W_q.bias.copy_(q_target)
        # Make diag code 2's embedding align strongly with that query
        # direction; every other code gets a small, orthogonal embedding.
        model.embeddings[0].weight.zero_()
        model.embeddings[0].weight[2, 0] = 10.0
        for code in (0, 1, 3):
            model.embeddings[0].weight[code, 1] = 1.0

    single_visit = _with_hadm([[[0, 1, 2, 3], [0]]], hadm_ids)
    with torch.no_grad():
        model(single_visit)

    attn = model.last_attention["diag"]
    codes = [0, 1, 2, 3]
    planted_index = codes.index(2)
    assert attn[planted_index] == pytest.approx(attn.max())
    assert attn[planted_index] > 0.9


def test_s3_lab_path_runs_and_changes_query():
    model = build_tiny_model(variant="S3", emb_dim=8)
    hadm_ids = [3001]
    model.load_text_features(
        hadm_ids=hadm_ids, emb=np.zeros((1, 768), dtype=np.float32), has_text=np.array([1], dtype=np.int8),
    )
    model.load_lab_features(hadm_ids=hadm_ids, bins=np.zeros((1, 30), dtype=np.int64))
    q_low = model._visit_query(3001).clone()

    model.load_lab_features(hadm_ids=hadm_ids, bins=np.full((1, 30), 5, dtype=np.int64))
    q_high = model._visit_query(3001).clone()

    assert not torch.allclose(q_low, q_high)


def test_s3_missing_lab_row_falls_back_to_missing_bin():
    model = build_tiny_model(variant="S3", emb_dim=8)
    hadm_ids = [3001]
    model.load_text_features(
        hadm_ids=hadm_ids, emb=np.zeros((1, 768), dtype=np.float32), has_text=np.array([1], dtype=np.int8),
    )
    model.load_lab_features(hadm_ids=[9999], bins=np.zeros((1, 30), dtype=np.int64))  # 3001 absent
    q = model._visit_query(3001)
    assert q.shape == (1, 8)  # runs without KeyError


def test_novelty_changes_output_only_for_s2_not_s1():
    hadm_ids = [4001]
    text_emb = np.random.RandomState(1).randn(1, 768).astype(np.float32)
    sample = _with_hadm([[[0, 1, 2], [0]]], hadm_ids)

    s1 = build_tiny_model(variant="S1", emb_dim=8)
    s1.load_text_features(hadm_ids=hadm_ids, emb=text_emb, has_text=np.array([1], dtype=np.int8))
    s1.novelty_by_hadm = {4001: {"diag": np.array([0, 0, 0], dtype=np.int8), "proc": np.array([0], dtype=np.int8)}}
    with torch.no_grad():
        out_s1_a, _ = s1(sample)
    s1.novelty_by_hadm = {4001: {"diag": np.array([2, 0, 1], dtype=np.int8), "proc": np.array([2], dtype=np.int8)}}
    with torch.no_grad():
        out_s1_b, _ = s1(sample)
    assert torch.allclose(out_s1_a, out_s1_b, atol=1e-7)  # S1 never reads novelty_by_hadm

    s2 = build_tiny_model(variant="S2", emb_dim=8)
    s2.load_state_dict(s1.state_dict(), strict=False)
    s2.load_text_features(hadm_ids=hadm_ids, emb=text_emb, has_text=np.array([1], dtype=np.int8))
    s2.load_novelty_features({4001: {"diag": np.array([0, 0, 0], dtype=np.int8), "proc": np.array([0], dtype=np.int8)}})
    with torch.no_grad():
        out_s2_a, _ = s2(sample)
    s2.load_novelty_features({4001: {"diag": np.array([2, 0, 1], dtype=np.int8), "proc": np.array([2], dtype=np.int8)}})
    with torch.no_grad():
        out_s2_b, _ = s2(sample)
    assert not torch.allclose(out_s2_a, out_s2_b, atol=1e-7)


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        build_tiny_model(variant="S9")


def test_augment_visits_with_hadm_appends_fourth_element_without_mutating_input():
    from safedrug_acute_attention_train import augment_visits_with_hadm

    data_split = [
        [[[0, 1], [0], [2]], [[1], [0, 1], [3]]],  # patient 0: 2 visits
        [[[2], [], [1]]],                            # patient 1: 1 visit
    ]
    hadm_ids = [[100, 101], [200]]
    out = augment_visits_with_hadm(data_split, patient_offset=0, hadm_ids=hadm_ids)

    assert out[0][0] == [[0, 1], [0], [2], 100]
    assert out[0][1] == [[1], [0, 1], [3], 101]
    assert out[1][0] == [[2], [], [1], 200]
    assert len(data_split[0][0]) == 3  # original untouched


def test_augment_visits_with_hadm_respects_patient_offset():
    from safedrug_acute_attention_train import augment_visits_with_hadm

    data_split = [[[[0], [0], [0]]]]
    hadm_ids = [[9999], [9999], [777]]  # patient_offset=2 -> use hadm_ids[2]
    out = augment_visits_with_hadm(data_split, patient_offset=2, hadm_ids=hadm_ids)
    assert out[0][0][3] == 777


def test_load_features_for_variant_s0_needs_nothing(tmp_path):
    from safedrug_acute_attention_train import load_features_for_variant

    assert load_features_for_variant("S0", tmp_path) == {}


def test_load_features_for_variant_s1_missing_text_raises(tmp_path):
    from safedrug_acute_attention_train import load_features_for_variant

    with pytest.raises(FileNotFoundError):
        load_features_for_variant("S1", tmp_path)


def test_load_features_for_variant_s1_loads_text(tmp_path):
    from safedrug_acute_attention_train import load_features_for_variant

    np.savez(
        tmp_path / "text_query.npz",
        HADM_ID=np.array([1, 2], dtype=np.int64),
        emb=np.zeros((2, 768), dtype=np.float32),
        has_text=np.array([1, 0], dtype=np.int8),
        text=np.array(["a", ""], dtype=object),
    )
    features = load_features_for_variant("S1", tmp_path)
    assert list(features["text_hadm_ids"]) == [1, 2]
    assert features["text_emb"].shape == (2, 768)
    assert "novelty" not in features


def test_collect_attention_weights_uses_stub_model_and_tags_split():
    from safedrug_acute_attention_train import collect_attention_weights

    class StubModel:
        def __init__(self):
            self.last_attention = {}
            self.calls = 0

        def eval(self):
            pass

        def __call__(self, seq_input):
            self.calls += 1
            n_diag = len(seq_input[-1][0])
            n_proc = len(seq_input[-1][1])
            self.last_attention = {
                "diag": np.full(n_diag, 1.0 / n_diag, dtype=np.float32),
                "proc": np.full(n_proc, 1.0 / n_proc, dtype=np.float32),
            }
            return None, None

    model = StubModel()
    data_split = [[[[0, 1], [0], [9], 555]]]  # 1 patient, 1 visit, hadm=555
    hadm_lookup = {(0, 0): (555, 42)}
    hadm_ids = [[555]]
    out = collect_attention_weights(model, data_split, "eval", hadm_lookup, hadm_ids, patient_offset=0)
    assert model.calls == 1
    assert out[555]["diag_codes"] == [0, 1]
    assert out[555]["diag_attn"] == pytest.approx([0.5, 0.5])
    assert out[555]["proc_codes"] == [0]
    assert out[555]["split"] == "eval"


def test_collect_attention_weights_raises_on_hadm_mismatch():
    from safedrug_acute_attention_train import collect_attention_weights

    class StubModel:
        last_attention = {"diag": np.array([1.0]), "proc": np.array([1.0])}

        def eval(self):
            pass

        def __call__(self, seq_input):
            return None, None

    data_split = [[[[0], [0], [0], 111]]]
    hadm_lookup = {(0, 0): (111, 1)}
    hadm_ids = [[999]]  # deliberately wrong -- must not match hadm_lookup's 111
    with pytest.raises(ValueError):
        collect_attention_weights(StubModel(), data_split, "test", hadm_lookup, hadm_ids, patient_offset=0)


# =========================================================== v2 attention (attn_version=2)

def test_v1_is_the_default_attn_version():
    model = build_tiny_model(variant="S1", emb_dim=8)
    assert model.attn_version == 1
    assert not hasattr(model, "LN_q")  # v2-only modules never built for the default


def test_unknown_attn_version_rejected():
    with pytest.raises(ValueError):
        build_tiny_model(variant="S1", emb_dim=8, attn_version=3)


def test_s0_ignores_attn_version_2_no_v2_modules_no_attention():
    model = build_tiny_model(variant="S0", emb_dim=8, attn_version=2)
    assert model.attn_version == 2
    assert not hasattr(model, "LN_q")  # S0 never needs the v2 modules
    with torch.no_grad():
        model(SAMPLE_INPUT_3)
    assert model.last_attention == {}  # S0 forward branch never calls _pool_modality


def test_v2_zero_query_gives_uniform_attention_and_matches_v1_s0_sum_bitwise():
    # q forced to zero (zero W_q weight+bias, zero no_text_embedding, and
    # has_text=False so the no_text fallback is actually used) must give
    # uniform attention and, since g=0 at init, i_m identical to the plain
    # v1-S0 sum of the (dropout'd) code embeddings -- bitwise.
    model = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    with torch.no_grad():
        model.W_q.weight.zero_()
        model.W_q.bias.zero_()
        model.no_text_embedding.zero_()
    model.load_text_features(
        hadm_ids=[5001], emb=np.zeros((1, 768), dtype=np.float32),
        has_text=np.array([0], dtype=np.int8),
    )
    q = model._visit_query(5001)
    assert torch.allclose(q, torch.zeros(1, 8), atol=1e-7)

    codes = [0, 1, 2, 3]
    expected_sum = (
        model.dropout(model.embeddings[0](torch.LongTensor(codes).unsqueeze(0)))
        .sum(dim=1)
        .squeeze(0)
    )

    with torch.no_grad():
        pooled = model._pool_modality(codes, DIAG, None, q)
    attn = model.last_attention["diag"]

    assert np.allclose(attn, np.full(len(codes), 1.0 / len(codes)), atol=1e-6)
    assert torch.allclose(pooled.squeeze(0).squeeze(0), expected_sum, atol=1e-6)


def test_v2_sharpens_where_v1_cannot_on_realistic_small_norms():
    # Reproduces the design brief's diagnosed defect at realistic production
    # scale (query norm ~2.0, key/embedding norm ~0.46, per the "Why"
    # section): v1's raw dot-product logits are too small (|logit| <~ 0.33
    # here) to concentrate softmax mass on any one code, while v2's
    # LayerNorm'd q/k normalizes that scale away and sharpens cleanly.
    q_c, k_c = 1.414, 0.325
    codes = [0, 1, 2, 3]
    planted_code = 2
    hadm_ids = [6001]
    single_visit = _with_hadm([[codes, [0]]], hadm_ids)

    def _plant(model):
        with torch.no_grad():
            model.W_q.weight.zero_()
            bias = torch.zeros(8)
            bias[0], bias[1] = q_c, -q_c
            model.W_q.bias.copy_(bias)
            model.embeddings[0].weight.zero_()
            for code in codes:
                sign = 1.0 if code == planted_code else -1.0
                model.embeddings[0].weight[code, 0] = sign * k_c
                model.embeddings[0].weight[code, 1] = -sign * k_c

    def _run(attn_version):
        model = build_tiny_model(variant="S1", emb_dim=8, attn_version=attn_version)
        model.load_text_features(
            hadm_ids=hadm_ids, emb=np.zeros((1, 768), dtype=np.float32),
            has_text=np.array([1], dtype=np.int8),
        )
        _plant(model)
        if attn_version == 2:
            # Make W_k an identity map and leave LN_k at its default
            # (weight=1, bias=0), so LayerNorm alone does the normalizing
            # work under test.
            with torch.no_grad():
                model.W_k[0].weight.copy_(torch.eye(8))
                model.W_k[0].bias.zero_()
        with torch.no_grad():
            model(single_visit)
        return model.last_attention["diag"]

    attn_v1 = _run(1)
    attn_v2 = _run(2)
    planted_idx = codes.index(planted_code)

    assert attn_v1[planted_idx] < 0.5  # v1: cannot sharpen at this scale
    assert attn_v2[planted_idx] == pytest.approx(attn_v2.max())
    assert attn_v2[planted_idx] > 0.9  # v2: sharpens via LayerNorm


def test_v2_attention_weights_sum_to_one_per_modality():
    model = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    hadm_ids = [7101, 7102]
    model.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(3).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    sample = _with_hadm(SAMPLE_INPUT_3, hadm_ids)
    with torch.no_grad():
        model(sample)
    assert model.last_attention["diag"].shape == (3,)  # visit 1 has 3 diag codes
    assert model.last_attention["proc"].shape == (2,)
    assert np.isclose(model.last_attention["diag"].sum(), 1.0, atol=1e-5)
    assert np.isclose(model.last_attention["proc"].sum(), 1.0, atol=1e-5)


def test_v2_pooled_scales_with_repeated_code_count():
    # A visit with the same code repeated n times has n identical keys, so
    # attention is exactly uniform regardless of q; pooled_m = n * (uniform
    # average of n identical embeddings) = n * that single embedding. With
    # g=0 at init the residual is zero, so this holds for i_m exactly, not
    # just approximately.
    model = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    model.load_text_features(
        hadm_ids=[7001], emb=np.random.RandomState(2).randn(1, 768).astype(np.float32),
        has_text=np.array([1], dtype=np.int8),
    )
    q = model._visit_query(7001)

    with torch.no_grad():
        pooled_1 = model._pool_modality([2], DIAG, None, q)
    attn_1 = model.last_attention["diag"]
    assert attn_1.shape == (1,)
    assert np.isclose(attn_1.sum(), 1.0, atol=1e-6)

    n = 5
    with torch.no_grad():
        pooled_n = model._pool_modality([2] * n, DIAG, None, q)
    attn_n = model.last_attention["diag"]
    assert attn_n.shape == (n,)
    assert np.isclose(attn_n.sum(), 1.0, atol=1e-6)
    assert np.allclose(attn_n, np.full(n, 1.0 / n), atol=1e-6)  # identical keys -> exactly uniform

    norm_1 = pooled_1.norm().item()
    norm_n = pooled_n.norm().item()
    assert norm_n == pytest.approx(n * norm_1, rel=1e-4)


# =================================== v2 + S2/S3 (novelty keys, lab query) ===================================

def test_v2_s2_forward_runs_and_attention_sums_to_one():
    model = build_tiny_model(variant="S2", emb_dim=8, attn_version=2)
    hadm_ids = [8001, 8002]
    model.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(4).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    model.load_novelty_features({
        8001: {"diag": np.array([0, 1], dtype=np.int8), "proc": np.array([0], dtype=np.int8)},
        8002: {"diag": np.array([2, 0, 1], dtype=np.int8), "proc": np.array([1, 0], dtype=np.int8)},
    })
    sample = _with_hadm(SAMPLE_INPUT_3, hadm_ids)
    with torch.no_grad():
        model(sample)
    assert model.last_attention["diag"].shape == (3,)  # target visit (last) has 3 diag codes
    assert model.last_attention["proc"].shape == (2,)  # target visit (last) has 2 proc codes
    assert np.isclose(model.last_attention["diag"].sum(), 1.0, atol=1e-5)
    assert np.isclose(model.last_attention["proc"].sum(), 1.0, atol=1e-5)


def test_v2_s2_novelty_state_change_changes_output():
    hadm_ids = [8101]
    text_emb = np.random.RandomState(5).randn(1, 768).astype(np.float32)
    sample = _with_hadm([[[0, 1, 2], [0]]], hadm_ids)

    model = build_tiny_model(variant="S2", emb_dim=8, attn_version=2)
    model.load_text_features(hadm_ids=hadm_ids, emb=text_emb, has_text=np.array([1], dtype=np.int8))

    model.load_novelty_features(
        {8101: {"diag": np.array([0, 0, 0], dtype=np.int8), "proc": np.array([0], dtype=np.int8)}}
    )
    with torch.no_grad():
        out_a, _ = model(sample)

    # Flip only diag code 2's novelty state (0 -> 2); everything else held fixed.
    model.load_novelty_features(
        {8101: {"diag": np.array([0, 0, 2], dtype=np.int8), "proc": np.array([0], dtype=np.int8)}}
    )
    with torch.no_grad():
        out_b, _ = model(sample)

    assert not torch.allclose(out_a, out_b, atol=1e-7)


def test_v2_s3_forward_runs_and_attention_sums_to_one():
    model = build_tiny_model(variant="S3", emb_dim=8, attn_version=2)
    hadm_ids = [9001, 9002]
    model.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(6).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    model.load_lab_features(
        hadm_ids=hadm_ids,
        bins=np.random.RandomState(7).randint(0, 6, size=(2, 30)).astype(np.int64),
    )
    sample = _with_hadm(SAMPLE_INPUT_3, hadm_ids)
    with torch.no_grad():
        model(sample)
    assert model.last_attention["diag"].shape == (3,)  # target visit (last) has 3 diag codes
    assert model.last_attention["proc"].shape == (2,)  # target visit (last) has 2 proc codes
    assert np.isclose(model.last_attention["diag"].sum(), 1.0, atol=1e-5)
    assert np.isclose(model.last_attention["proc"].sum(), 1.0, atol=1e-5)


def test_v2_s3_lab_bin_change_changes_output():
    model = build_tiny_model(variant="S3", emb_dim=8, attn_version=2)
    hadm_ids = [9101]
    model.load_text_features(
        hadm_ids=hadm_ids, emb=np.zeros((1, 768), dtype=np.float32), has_text=np.array([1], dtype=np.int8),
    )
    sample = _with_hadm([[[0, 1, 2], [0]]], hadm_ids)

    model.load_lab_features(hadm_ids=hadm_ids, bins=np.zeros((1, 30), dtype=np.int64))
    with torch.no_grad():
        out_low, _ = model(sample)

    model.load_lab_features(hadm_ids=hadm_ids, bins=np.full((1, 30), 5, dtype=np.int64))
    with torch.no_grad():
        out_high, _ = model(sample)

    assert not torch.allclose(out_low, out_high, atol=1e-7)


def test_v2_s3_missing_lab_row_falls_back_without_error():
    model = build_tiny_model(variant="S3", emb_dim=8, attn_version=2)
    hadm_ids = [9201]
    model.load_text_features(
        hadm_ids=hadm_ids, emb=np.zeros((1, 768), dtype=np.float32), has_text=np.array([1], dtype=np.int8),
    )
    model.load_lab_features(hadm_ids=[9999], bins=np.zeros((1, 30), dtype=np.int64))  # 9201 absent
    sample = _with_hadm([[[0, 1, 2], [0]]], hadm_ids)
    with torch.no_grad():
        out, _ = model(sample)
    assert out.shape == (1, 4)  # n_med=4, runs without KeyError -- falls back to the missing bin
    assert model.last_attention["diag"].shape == (3,)
    assert model.last_attention["proc"].shape == (1,)


def test_parse_args_attn_version_defaults_to_one():
    from safedrug_acute_attention_train import parse_args

    args = parse_args(["--variant", "S0", "--features-dir", "x", "--out-dir", "y"])
    assert args.attn_version == 1


def test_parse_args_accepts_attn_version_two():
    from safedrug_acute_attention_train import parse_args

    args = parse_args(
        ["--variant", "S1", "--features-dir", "x", "--out-dir", "y", "--attn-version", "2"]
    )
    assert args.attn_version == 2


def test_parse_args_rejects_invalid_attn_version():
    from safedrug_acute_attention_train import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--variant", "S0", "--features-dir", "x", "--out-dir", "y", "--attn-version", "3"])


def _fake_build_mpnn(molecule, idx2word, radius, device):
    _vocab, _ddi_adj, _ddi_mask_H, MPNNSet, N_fingerprint, average_projection = _tiny_ctor_args(
        n_med=len(idx2word)
    )
    return MPNNSet, N_fingerprint, average_projection


def test_build_model_threads_attn_version_into_the_model(monkeypatch):
    import safedrug_acute_attention_train as train_mod

    monkeypatch.setattr(train_mod, "buildMPNN", _fake_build_mpnn)

    voc_size, ddi_adj, ddi_mask_H, _MPNNSet, _N_fp, _avg_proj = _tiny_ctor_args(
        n_diag=6, n_proc=5, n_med=4
    )
    med_voc = types.SimpleNamespace(idx2word=list(range(4)))
    device = torch.device("cpu")
    features = {
        "text_hadm_ids": [1],
        "text_emb": np.zeros((1, 768), dtype=np.float32),
        "has_text": np.array([1], dtype=np.int8),
    }

    model_v2 = train_mod.build_model(
        "S1", voc_size, ddi_adj, ddi_mask_H, None, med_voc, device, features, attn_version=2,
    )
    assert model_v2.attn_version == 2
    assert hasattr(model_v2, "LN_q")  # v2-only module actually built

    model_v1 = train_mod.build_model(
        "S1", voc_size, ddi_adj, ddi_mask_H, None, med_voc, device, features,
    )
    assert model_v1.attn_version == 1  # default, unchanged
    assert not hasattr(model_v1, "LN_q")


def test_build_model_s0_ignores_attn_version(monkeypatch):
    import safedrug_acute_attention_train as train_mod

    monkeypatch.setattr(train_mod, "buildMPNN", _fake_build_mpnn)

    voc_size, ddi_adj, ddi_mask_H, _MPNNSet, _N_fp, _avg_proj = _tiny_ctor_args(
        n_diag=6, n_proc=5, n_med=4
    )
    med_voc = types.SimpleNamespace(idx2word=list(range(4)))
    device = torch.device("cpu")

    model = train_mod.build_model(
        "S0", voc_size, ddi_adj, ddi_mask_H, None, med_voc, device, {}, attn_version=2,
    )
    assert model.variant == "S0"
    assert not hasattr(model, "LN_q")  # --variant S0 ignores attn_version: no v2 modules built

    with torch.no_grad():
        model(SAMPLE_INPUT_3)
    assert model.last_attention == {}  # S0 path runs regardless of attn_version


# ============================================== supervised acute attention (S2s)

def test_last_attention_tensor_populated_only_for_attn_version_2_and_has_grad():
    hadm_ids = [10001, 10002]

    model_v1 = build_tiny_model(variant="S1", emb_dim=8, attn_version=1)
    model_v1.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(11).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    sample = _with_hadm(SAMPLE_INPUT_3, hadm_ids)
    model_v1(sample)  # no torch.no_grad() -- exercising autograd tracking too
    assert model_v1.last_attention_tensor == {}  # v1 never populates it

    model_v2 = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    model_v2.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(12).randn(2, 768).astype(np.float32),
        has_text=np.array([1, 1], dtype=np.int8),
    )
    model_v2(sample)

    tensor_entry = model_v2.last_attention_tensor
    assert set(tensor_entry.keys()) == {"diag", "diag_codes"}
    a_diag = tensor_entry["diag"]
    assert a_diag.requires_grad
    assert torch.isclose(a_diag.sum(), torch.tensor(1.0), atol=1e-5)
    # SAMPLE_INPUT_3's last (target) visit is [[1, 2, 3], [0, 1]] -- diag_codes
    # must match its diag code list, in order.
    assert tensor_entry["diag_codes"] == list(SAMPLE_INPUT_3[-1][0])
    np.testing.assert_allclose(
        a_diag.detach().cpu().numpy(), model_v2.last_attention["diag"], atol=1e-6
    )


def test_last_attention_tensor_never_populated_for_proc_modality():
    model = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    hadm_ids = [10101]
    model.load_text_features(
        hadm_ids=hadm_ids,
        emb=np.random.RandomState(13).randn(1, 768).astype(np.float32),
        has_text=np.array([1], dtype=np.int8),
    )
    sample = _with_hadm([[[0, 1, 2], [0, 1]]], hadm_ids)
    with torch.no_grad():
        model(sample)
    assert "proc" not in model.last_attention_tensor
    assert model.last_attention_tensor["diag_codes"] == [0, 1, 2]


def test_s0_never_populates_last_attention_tensor_even_with_attn_version_2():
    model = build_tiny_model(variant="S0", emb_dim=8, attn_version=2)
    with torch.no_grad():
        model(SAMPLE_INPUT_3)
    assert model.last_attention_tensor == {}


# ------------------------------------------------- target mapping (HADM_ID -> positions)

def test_load_seq1_map_preserves_leading_zero_codes(tmp_path):
    from safedrug_acute_attention_train import load_seq1_map

    csv_path = tmp_path / "ccs.csv"
    csv_path.write_text(
        "SUBJECT_ID,HADM_ID,n_dx,seq1_code,seq1_name\n"
        "1,100,5,03811,Septicemia\n"
        "2,200,3,4280,CHF\n",
        encoding="utf-8",
    )
    seq1_map = load_seq1_map(csv_path)
    assert seq1_map == {100: "03811", 200: "4280"}  # leading zero NOT stripped


def test_build_attn_supervision_targets_planted_missing_oov_and_not_in_visit():
    from safedrug_acute_attention_train import build_attn_supervision_targets

    diag_voc = types.SimpleNamespace(word2idx={"4280": 0, "5849": 1, "99591": 2})
    data_train = [
        # patient 0, hadm=100: seq1 "4280" -> vocab idx 0 -> position 1 in [1,0,2]
        [[[1, 0, 2], [0], [0], 100]],
        # patient 1, hadm=200: seq1_map has no entry at all for 200 -> missing
        [[[1, 0, 2], [0], [0], 200]],
        # patient 2, hadm=300: seq1 "99999" not in diag_voc.word2idx -> out-of-vocab
        [[[1, 0, 2], [0], [0], 300]],
        # patient 3, hadm=400: seq1 "5849" -> vocab idx 1, but 1 is not in adm[0]=[0,2]
        [[[0, 2], [0], [0], 400]],
    ]
    seq1_map = {100: "4280", 300: "99999", 400: "5849"}

    targets = build_attn_supervision_targets(data_train, seq1_map, diag_voc)

    assert targets == {100: [1]}


def test_build_attn_supervision_targets_returns_all_positions_for_repeated_code():
    from safedrug_acute_attention_train import build_attn_supervision_targets

    diag_voc = types.SimpleNamespace(word2idx={"4280": 5})
    data_train = [[[[5, 1, 5], [0], [0], 900]]]
    seq1_map = {900: "4280"}

    targets = build_attn_supervision_targets(data_train, seq1_map, diag_voc)

    assert targets == {900: [0, 2]}


# --------------------------------------------------------------- auxiliary loss

def test_attn_aux_loss_equals_neg_log_mass_on_targets():
    from safedrug_acute_attention_train import attn_aux_loss

    attn = torch.tensor([0.1, 0.6, 0.3], requires_grad=True)
    positions = [0, 2]
    loss = attn_aux_loss(attn, positions)
    expected = -math.log(0.1 + 0.3 + 1e-8)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert loss.requires_grad


class _StubAttnModel(torch.nn.Module):
    """Minimal stand-in for SafeDrugAcuteAttention exposing exactly what
    train_one_epoch_supervised touches: .train() (inherited from
    nn.Module), .tensor_ddi_adj, forward(seq_input) -> (result, loss_ddi),
    and .last_attention_tensor["diag"] -- populated on every forward()
    call from its own real nn.Parameter (attn_logits) via softmax, so an
    optimizer step can actually move mass toward a target position.
    Deterministic zero init (no dropout, no random weights), following the
    _StubDDIModel pattern in tests/test_safedrug_train_dump.py."""

    def __init__(self, ddi_adj: np.ndarray, n_med: int, n_diag: int):
        super().__init__()
        self.tensor_ddi_adj = torch.tensor(ddi_adj, dtype=torch.float32)
        self.w = torch.nn.Parameter(torch.zeros(n_med))
        self.attn_logits = torch.nn.Parameter(torch.zeros(n_diag))
        self.last_attention_tensor = {}
        self.forward_calls = 0

    def forward(self, seq_input):
        self.forward_calls += 1
        result = self.w.unsqueeze(0)  # (1, n_med)
        prob = torch.sigmoid(result)
        neg_pred_prob = prob.t() * prob
        loss_ddi = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        attn = torch.softmax(self.attn_logits, dim=0)
        self.last_attention_tensor = {"diag": attn, "diag_codes": list(seq_input[-1][0])}
        return result, loss_ddi


def test_train_one_epoch_supervised_mu_zero_matches_unsupervised(tmp_path):
    from safedrug_acute_attention_train import train_one_epoch_supervised
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    data_train = [[[[0, 1, 2], [], [0, 1], 555]]]  # 1 patient, 1 visit, meds {0,1}

    model_plain = _StubAttnModel(ddi_adj, n_med=2, n_diag=3)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=1e-2)
    loss_plain = train_one_epoch(
        model_plain, data_train, opt_plain, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05,
    )

    model_sup = _StubAttnModel(ddi_adj, n_med=2, n_diag=3)  # identical deterministic zero init
    opt_sup = torch.optim.Adam(model_sup.parameters(), lr=1e-2)
    loss_sup, aux_loss_mean = train_one_epoch_supervised(
        model_sup, data_train, opt_sup, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, attn_targets={555: [1]}, mu=0.0,
    )

    assert loss_sup == pytest.approx(loss_plain)
    assert aux_loss_mean != pytest.approx(0.0)  # aux loss WAS computed (just zero-weighted)
    assert torch.allclose(model_plain.w, model_sup.w)  # identical parameter update


def test_train_one_epoch_supervised_no_targets_matches_unsupervised(tmp_path):
    # Default-path guard: an empty attn_targets dict (e.g. --attn-supervision
    # none never builds one) must behave exactly like plain train_one_epoch
    # regardless of mu.
    from safedrug_acute_attention_train import train_one_epoch_supervised
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    data_train = [[[[0, 1, 2], [], [0, 1], 555]]]

    model_plain = _StubAttnModel(ddi_adj, n_med=2, n_diag=3)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=1e-2)
    loss_plain = train_one_epoch(
        model_plain, data_train, opt_plain, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05,
    )

    model_sup = _StubAttnModel(ddi_adj, n_med=2, n_diag=3)
    opt_sup = torch.optim.Adam(model_sup.parameters(), lr=1e-2)
    loss_sup, aux_loss_mean = train_one_epoch_supervised(
        model_sup, data_train, opt_sup, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, attn_targets={}, mu=5.0,
    )

    assert loss_sup == pytest.approx(loss_plain)
    assert aux_loss_mean == pytest.approx(0.0)  # no visit had a target
    assert torch.allclose(model_plain.w, model_sup.w)


def test_train_one_epoch_supervised_positive_mu_increases_target_attention_mass_stub(tmp_path):
    from safedrug_acute_attention_train import train_one_epoch_supervised

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    data_train = [[[[0, 1, 2], [], [0, 1], 555]]]
    target_position = 1

    model = _StubAttnModel(ddi_adj, n_med=2, n_diag=3)
    before_mass = torch.softmax(model.attn_logits.detach(), dim=0)[target_position].item()
    assert before_mass == pytest.approx(1 / 3)  # uniform at zero init

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_one_epoch_supervised(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, attn_targets={555: [target_position]}, mu=5.0,
    )
    after_mass = torch.softmax(model.attn_logits.detach(), dim=0)[target_position].item()

    assert after_mass > before_mass


def test_train_one_epoch_supervised_positive_mu_increases_target_attention_mass_real_model(tmp_path):
    # End-to-end integration on the real tiny SafeDrugAcuteAttention (S1,
    # attn_version=2): one gradient step with mu > 0 must increase the
    # diag attention mass on the seq1-mapped target position.
    from safedrug_acute_attention_train import train_one_epoch_supervised

    model = build_tiny_model(variant="S1", emb_dim=8, attn_version=2)
    hadm_id = 30001
    model.load_text_features(
        hadm_ids=[hadm_id],
        emb=np.random.RandomState(30).randn(1, 768).astype(np.float32),
        has_text=np.array([1], dtype=np.int8),
    )
    diag_codes = [0, 1, 2, 3]
    data_train = [[[diag_codes, [0], [0, 1], hadm_id]]]  # n_med=4 (tiny ctor default) -> meds {0,1} valid
    ddi_adj = np.zeros((4, 4))
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    target_position = 2  # diag_codes[2] == 2
    attn_targets = {hadm_id: [target_position]}

    with torch.no_grad():
        model(data_train[0])
    before_mass = model.last_attention_tensor["diag"][target_position].item()

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_one_epoch_supervised(
        model, data_train, optimizer, torch.device("cpu"), (6, 5, 4), ddi_adj_path,
        target_ddi=0.06, kp=0.05, attn_targets=attn_targets, mu=50.0,
    )

    with torch.no_grad():
        model(data_train[0])
    after_mass = model.last_attention_tensor["diag"][target_position].item()

    assert after_mass > before_mass


def test_parse_args_attn_supervision_defaults_to_none():
    from safedrug_acute_attention_train import parse_args

    args = parse_args(["--variant", "S1", "--features-dir", "x", "--out-dir", "y"])
    assert args.attn_supervision == "none"
    assert args.attn_sup_weight == 1.0
    assert args.ccs_csv.endswith("ccs_assignments.csv")


def test_parse_args_accepts_attn_supervision_seq1_and_custom_weight():
    from safedrug_acute_attention_train import parse_args

    args = parse_args([
        "--variant", "S1", "--features-dir", "x", "--out-dir", "y",
        "--attn-supervision", "seq1", "--attn-sup-weight", "2.5", "--ccs-csv", "custom.csv",
    ])
    assert args.attn_supervision == "seq1"
    assert args.attn_sup_weight == 2.5
    assert args.ccs_csv == "custom.csv"


def test_parse_args_rejects_invalid_attn_supervision():
    from safedrug_acute_attention_train import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--variant", "S1", "--features-dir", "x", "--out-dir", "y",
            "--attn-supervision", "bogus",
        ])
