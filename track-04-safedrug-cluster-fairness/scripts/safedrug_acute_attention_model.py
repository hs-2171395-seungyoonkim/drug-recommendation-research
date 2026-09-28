"""SafeDrug acute-attention visit-pooling model (R4).

Subclasses SOTA/SafeDrug/src/models.py's real SafeDrugModel, replacing only
the two sum-poolings (models.py lines 200-227) with single-head attention
pooling. Everything from `o1, h1 = self.encoders[0](i1_seq)` onward
(models.py lines 229-252) is reproduced functionally identical (verified by
test) in this class's own forward -- that method has no seam to call into
partially, so it is copied rather than inherited.

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R4") for the full contract and rationale.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")

if "dnc" not in sys.modules:
    _dnc_stub = types.ModuleType("dnc")
    _dnc_stub.DNC = object
    sys.modules["dnc"] = _dnc_stub
if str(SAFEDRUG_SRC) not in sys.path:
    sys.path.insert(0, str(SAFEDRUG_SRC))

from models import SafeDrugModel  # noqa: E402  (needs the dnc stub + sys.path insert above)

VALID_VARIANTS = ("S0", "S1", "S2", "S3")
DIAG, PROC = 0, 1


class SafeDrugAcuteAttention(SafeDrugModel):
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=64,
        device=torch.device("cpu:0"),
        variant="S0",
        text_dim=768,
        n_lab_items=30,
        n_lab_bins=6,
        lab_bin_dim=8,
        attn_version=1,
    ):
        super().__init__(
            vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints,
            average_projection, emb_dim=emb_dim, device=device,
        )
        if variant not in VALID_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {VALID_VARIANTS}")
        if attn_version not in (1, 2):
            raise ValueError(f"unknown attn_version {attn_version!r}, expected 1 or 2")
        self.variant = variant
        self.attn_version = attn_version
        self.emb_dim = emb_dim
        self.text_dim = text_dim
        self.n_lab_items = n_lab_items
        self.n_lab_bins = n_lab_bins
        self.lab_bin_dim = lab_bin_dim

        # HADM_ID -> row-index lookups. Plain python dicts, not buffers: an
        # nn.Module buffer must be a Tensor, and these never need .to(device)
        # movement or a gradient of their own.
        self.text_row_of_hadm: dict[int, int] = {}
        self.lab_row_of_hadm: dict[int, int] = {}
        # Ragged per-visit novelty arrays, keyed directly by HADM_ID (R2's
        # contract: int8 arrays positionally aligned with that visit's own
        # diag/proc code list in records_final -- never a fixed-width
        # matrix, so this cannot be a buffer either).
        self.novelty_by_hadm: dict[int, dict[str, np.ndarray]] = {}
        # Populated by _pool_modality on every S1/S2/S3 forward call;
        # overwritten per visit, so after one forward() call it holds
        # exactly the newest (target) visit's per-modality attention
        # weights. Never populated for S0 (there is no attention).
        self.last_attention: dict[str, np.ndarray] = {}
        # DIFFERENTIABLE counterpart of last_attention["diag"], populated
        # only in the v2 pooling path (attn_version == 2, S1/S2/S3) --
        # never for v1 or S0 (stays {} there), and never for the "proc"
        # modality (supervision only ever targets a diagnosis code).
        # "diag": the softmax attention tensor itself (still attached to
        # the autograd graph, NOT detached); "diag_codes": the same-order
        # python list of diag code indices _pool_modality was called with
        # for that visit (== that visit's adm[0]), so a caller can map a
        # vocabulary index to a position in this tensor. Overwritten per
        # visit exactly like last_attention.
        self.last_attention_tensor: dict[str, object] = {}

        if variant in ("S1", "S2", "S3"):
            self.W_q = nn.Linear(text_dim, emb_dim)
            self.no_text_embedding = nn.Parameter(torch.zeros(text_dim))
            nn.init.normal_(self.no_text_embedding, std=0.02)
            self.W_r = nn.Linear(emb_dim, emb_dim)
            # Placeholder buffers (0 rows) -- load_text_features() replaces
            # them with the real, loaded arrays before any real forward
            # call; the "missing text" fallback path is exercised via
            # has_text/absent-HADM_ID, not via an empty buffer.
            self.register_buffer("text_emb", torch.zeros(0, text_dim), persistent=False)
            self.register_buffer("has_text", torch.zeros(0, dtype=torch.bool), persistent=False)

        if variant in ("S2", "S3"):
            self.novelty_embeddings = nn.ModuleList(
                [nn.Embedding(3, emb_dim) for _ in range(2)]  # [DIAG]=0, [PROC]=1
            )

        if variant == "S3":
            self.lab_item_embeddings = nn.ModuleList(
                [nn.Embedding(n_lab_bins, lab_bin_dim) for _ in range(n_lab_items)]
            )
            self.W_lab = nn.Linear(n_lab_items * lab_bin_dim, emb_dim)
            self.register_buffer(
                "lab_bins", torch.zeros(0, n_lab_items, dtype=torch.long), persistent=False
            )

        if variant in ("S1", "S2", "S3") and attn_version == 2:
            # v2-only attention parameterisation (LayerNorm'd q/k, gated
            # residual). S0 never needs these -- its forward branch never
            # calls _visit_query/_pool_modality regardless of attn_version,
            # so `--variant S0 --attn-version 2` builds none of this and
            # is functionally plain S0.
            self.LN_q = nn.LayerNorm(emb_dim)  # shared across modalities
            self.W_k = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(2)])  # per modality
            self.LN_k = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(2)])  # per modality
            self.g = nn.Parameter(torch.zeros(1))  # shared across modalities; 0 at init

    # ------------------------------------------------------------- feature loading
    def load_text_features(self, hadm_ids, emb: np.ndarray, has_text: np.ndarray) -> None:
        """hadm_ids: 1-D sequence of int HADM_IDs, row-aligned with
        emb/has_text (features/text_query.npz's own arrays). S1/S2/S3 only."""
        self.text_row_of_hadm = {int(h): i for i, h in enumerate(hadm_ids)}
        self.text_emb = torch.as_tensor(np.asarray(emb), dtype=torch.float32).to(self.device)
        self.has_text = torch.as_tensor(np.asarray(has_text).astype(bool)).to(self.device)

    def load_lab_features(self, hadm_ids, bins: np.ndarray) -> None:
        """hadm_ids: 1-D sequence of int HADM_IDs, row-aligned with bins
        (features/lab_query.npz's own arrays). S3 only."""
        self.lab_row_of_hadm = {int(h): i for i, h in enumerate(hadm_ids)}
        self.lab_bins = torch.as_tensor(np.asarray(bins), dtype=torch.long).to(self.device)

    def load_novelty_features(self, novelty: dict) -> None:
        """novelty: features/novelty.pkl's own dict, HADM_ID -> {"diag":
        int8 array, "proc": int8 array}. S2/S3 only."""
        self.novelty_by_hadm = novelty

    # ------------------------------------------------------------- query construction
    def _visit_query(self, hadm_id: int) -> torch.Tensor:
        """(1, emb_dim) query for one visit. S1/S2: text only. S3: text +
        lab. Missing text (hadm_id absent from the loaded lookup, or
        has_text=False for its row) falls back to the learned
        no_text_embedding as the query input, per R1."""
        row = self.text_row_of_hadm.get(int(hadm_id))
        if row is not None and bool(self.has_text[row]):
            text_input = self.text_emb[row].unsqueeze(0)  # (1, text_dim)
        else:
            text_input = self.no_text_embedding.unsqueeze(0)  # (1, text_dim)
        q = self.W_q(text_input)  # (1, emb_dim)
        if self.variant == "S3":
            lab_row = self.lab_row_of_hadm.get(int(hadm_id))
            if lab_row is None:
                bins = torch.full(
                    (self.n_lab_items,), self.n_lab_bins - 1, dtype=torch.long, device=self.device,
                )
            else:
                bins = self.lab_bins[lab_row]
            lab_parts = [self.lab_item_embeddings[j](bins[j]) for j in range(self.n_lab_items)]
            lab_concat = torch.cat(lab_parts, dim=-1).unsqueeze(0)  # (1, n_lab_items*lab_bin_dim)
            q = q + self.W_lab(lab_concat)
        if self.attn_version == 2:
            # LN_q is shared across modalities and applied once, here, to
            # the SUM of the query terms -- both _pool_modality calls for
            # this visit (DIAG, PROC) reuse this same LayerNorm'd q.
            q = self.LN_q(q)
        return q

    # ------------------------------------------------------------- attention pooling
    def _pool_modality(self, codes, modality_idx: int, novelty_arr, q: torch.Tensor) -> torch.Tensor:
        """One modality (DIAG=0, PROC=1) of one visit. Returns (1,1,emb_dim)
        -- the same shape the original sum_embedding produced, so the
        i1_seq/i2_seq accumulation in forward() is unaffected by variant.
        Stores this visit's attention weights (numpy, in `codes` order) into
        self.last_attention, overwriting the previous visit's entry."""
        codes_t = torch.LongTensor(codes).unsqueeze(dim=0).to(self.device)  # (1, n)
        raw_emb = self.dropout(self.embeddings[modality_idx](codes_t))  # (1, n, dim)
        emb = raw_emb.squeeze(0)  # (n, dim)

        if self.attn_version == 2:
            n = emb.shape[0]
            key_input = emb
            if self.variant in ("S2", "S3") and novelty_arr is not None:
                novelty_t = torch.as_tensor(novelty_arr, dtype=torch.long, device=self.device)  # (n,)
                key_input = key_input + self.novelty_embeddings[modality_idx](novelty_t)
            keys = self.LN_k[modality_idx](self.W_k[modality_idx](key_input))  # (n, dim)

            scores = torch.mv(keys, q.squeeze(0)) / math.sqrt(self.emb_dim)  # (n,)
            attn = F.softmax(scores, dim=0)  # (n,)
            # VALUE uses the plain (dropout'd) code embedding, scaled by the
            # code count so uniform attention reproduces the original v1-S0
            # sum exactly; the residual is gated by the learnable scalar g
            # (zero at init -- v2 starts identical to that sum).
            pooled = n * torch.mv(emb.t(), attn) + self.g * self.W_r(q).squeeze(0)  # (dim,)
            if modality_idx == DIAG:
                # Keep the graph-attached tensor (no .detach()) so an
                # auxiliary loss built on it can backprop into W_q/W_k/etc.
                self.last_attention_tensor = {"diag": attn, "diag_codes": list(codes)}
        else:
            keys = emb
            if self.variant in ("S2", "S3") and novelty_arr is not None:
                novelty_t = torch.as_tensor(novelty_arr, dtype=torch.long, device=self.device)  # (n,)
                keys = keys + self.novelty_embeddings[modality_idx](novelty_t)

            scores = torch.mv(keys, q.squeeze(0)) / math.sqrt(self.emb_dim)  # (n,)
            attn = F.softmax(scores, dim=0)  # (n,)
            # The VALUE uses the plain code embedding only -- novelty affects
            # the attention weights, never the pooled representation directly.
            pooled = torch.mv(emb.t(), attn) + self.W_r(q).squeeze(0)  # (dim,)

        key = "diag" if modality_idx == DIAG else "proc"
        self.last_attention[key] = attn.detach().cpu().numpy()
        return pooled.unsqueeze(0).unsqueeze(0)  # (1,1,dim)

    # ------------------------------------------------------------- forward
    def forward(self, input):
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            if self.variant == "S0":
                # Functionally identical (verified by test) to models.py's
                # own pooling loop (lines 200-227) -- never routed through
                # _pool_modality.
                i1 = sum_embedding(
                    self.dropout(
                        self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))
                    )
                )
                i2 = sum_embedding(
                    self.dropout(
                        self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))
                    )
                )
            else:
                # HADM_ID is always the LAST element of adm for S1/S2/S3:
                # adm[3] in the real training pipeline (diag, proc, med
                # targets, then HADM_ID appended by augment_visits_with_hadm),
                # or adm[2]/adm[-1] in lighter unit-test fixtures that omit
                # the med-targets element -- forward() itself never reads
                # med targets, so indexing from the end is robust to both.
                hadm_id = adm[-1]
                q = self._visit_query(hadm_id)
                novelty = self.novelty_by_hadm.get(int(hadm_id)) if self.variant in ("S2", "S3") else None
                diag_novelty = novelty["diag"] if novelty is not None else None
                proc_novelty = novelty["proc"] if novelty is not None else None
                i1 = self._pool_modality(adm[0], DIAG, diag_novelty, q)
                i2 = self._pool_modality(adm[1], PROC, proc_novelty, q)
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        # ---- models.py lines 229-252, functionally identical (verified by test) from here on ----
        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)
        query = self.query(patient_representations)[-1:, :]

        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        )

        result = torch.mul(bipartite_emb, MPNN_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg
