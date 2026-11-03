"""OPA-DPO loss (paper Eq. 7-10).

Inputs are per-example lists of per-token log-probs (policy and reference pi_OPA) for
each role, plus token weights W_hal / W_img. A role entry may be None (no GT response).

loss_mode="paper"    : sequence-level DPO, log pi^w(y) = sum_i W_i log pi(y_i).
loss_mode="official" : what the released code computes (opadpo/dpo_models/dpo_trainer.py):
                       sigmoid applied per token position with chosen/rejected aligned by
                       index, averaged over a fixed padded length; the anchor term pulls the
                       GT reward up but the revised-response reward *down*, twice each.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import DPOConfig


def _wsum(lp: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    return (lp * w.to(lp)).sum() if w is not None else lp.sum()


def _pad(ts: list[torch.Tensor], n: int) -> torch.Tensor:
    return torch.stack([F.pad(t, (0, n - len(t))) for t in ts])


def _token_pair(c, c_ref, wc, r, r_ref, wr, beta: float, norm: int) -> torch.Tensor:
    """Token-level DPO term of the released code: sum_t -log sigma(beta * logit_t) / norm."""
    n = max(len(c), len(r))
    ones_c = torch.ones_like(c) if wc is None else wc.to(c)
    ones_r = torch.ones_like(r) if wr is None else wr.to(r)
    dc, dr = _pad([(c - c_ref) * ones_c], n)[0], _pad([(r - r_ref) * ones_r], n)[0]
    return -F.logsigmoid(beta * (dc - dr)).sum() / norm


def opa_dpo_loss(pol: dict, ref: dict, w: dict, cfg: DPOConfig) -> tuple[torch.Tensor, dict]:
    """pol/ref: role -> list (len B) of 1-D tensors or None. w: gen_hal/rev_hal/rev_img -> list of tensors."""
    beta, delta = cfg.beta, cfg.delta
    B = len(w["rev_hal"])
    losses, stats = [], {k: [] for k in ("lc", "if", "anc", "r_gt", "r_rev", "r_gen", "acc_rev_gen")}

    for b in range(B):
        has_gt = pol.get("gt_m") is not None and pol["gt_m"][b] is not None
        g = lambda role: (pol[role][b], ref[role][b])  # noqa: E731

        if cfg.loss_mode == "paper":
            def r(role, wt=None):
                p, q = g(role)
                return _wsum(p, wt) - _wsum(q.to(p), wt)

            r_gen, r_gen_h = r("gen_m"), r("gen_m", w["gen_hal"][b])
            r_rev, r_rev_h, r_rev_i = r("rev_m"), r("rev_m", w["rev_hal"][b]), r("rev_m", w["rev_img"][b])
            r_rev_mm_i = r("rev_mm", w["rev_img"][b])
            lc = -F.logsigmoid(beta * (r_rev_h - r_gen_h))
            if_ = -F.logsigmoid(beta * (r_rev_i - r_rev_mm_i))
            anc = -F.logsigmoid(beta * r_rev - delta)
            if has_gt:
                r_gt, r_gt_mm = r("gt_m"), r("gt_mm")
                lc = lc - F.logsigmoid(beta * (r_gt - r_gen))
                if_ = if_ - F.logsigmoid(beta * (r_gt - r_gt_mm))
                anc = anc - F.logsigmoid(beta * r_gt - delta)
                stats["r_gt"].append(beta * r_gt.detach())
        elif cfg.loss_mode == "official":
            norm = cfg.official_response_len
            (gen, gen_q), (rev, rev_q), (rev_mm, rev_mm_q) = g("gen_m"), g("rev_m"), g("rev_mm")
            gen_q, rev_q, rev_mm_q = gen_q.to(gen), rev_q.to(rev), rev_mm_q.to(rev_mm)
            lc = _token_pair(rev, rev_q, w["rev_hal"][b], gen, gen_q, w["gen_hal"][b], beta, norm)
            if_ = _token_pair(rev, rev_q, w["rev_img"][b], rev_mm, rev_mm_q, w["rev_img"][b], beta, norm)
            d_rev = beta * (rev - rev_q)
            anc = 2 * -F.logsigmoid(-d_rev + delta).sum() / norm
            r_gen, r_rev = (gen - gen_q).sum(), (rev - rev_q).sum()
            if has_gt:
                (gt, gt_q), (gt_mm, gt_mm_q) = g("gt_m"), g("gt_mm")
                gt_q, gt_mm_q = gt_q.to(gt), gt_mm_q.to(gt_mm)
                lc = lc + _token_pair(gt, gt_q, None, gen, gen_q, None, beta, norm)
                if_ = if_ + _token_pair(gt, gt_q, None, gt_mm, gt_mm_q, None, beta, norm)
                anc = anc + 2 * -F.logsigmoid(beta * (gt - gt_q) - delta).sum() / norm
                stats["r_gt"].append(beta * (gt - gt_q).sum().detach())
        else:
            raise ValueError(f"unknown loss_mode {cfg.loss_mode!r}")

        losses.append(lc + cfg.gamma_if * if_ + cfg.gamma_anc * anc)
        stats["lc"].append(lc.detach())
        stats["if"].append(if_.detach())
        stats["anc"].append(anc.detach())
        stats["r_rev"].append(beta * r_rev.detach())
        stats["r_gen"].append(beta * r_gen.detach())
        stats["acc_rev_gen"].append((r_rev > r_gen).float().detach())

    metrics = {k: torch.stack(v).mean().item() for k, v in stats.items() if v}
    return torch.stack(losses).mean(), metrics
