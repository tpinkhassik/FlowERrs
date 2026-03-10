"""
DiverseFlow: DPP-based diversity-promoting sampling for flow matching.

Implements the method from "DiverseFlow: Sample-Efficient Diverse Mode Coverage
in Flows" (Morshed & Boddeti, 2025, arXiv:2504.07894).

K samples per reaction are jointly evolved with a DPP repulsive term that
promotes diversity among predicted products while preserving chemical
constraints (electron conservation, BE matrix symmetry).
"""

import torch
import torchdiffeq


def _dpp_log_likelihood(x_groups, eps=1e-6):
    """
    Compute DPP log-likelihood: log det(L) - log det(L+I).

    Parameters
    ----------
    x_groups : Tensor, shape (num_reactions, K, D)
        Flattened sample estimates grouped by reaction.

    Returns
    -------
    Scalar: sum of log-likelihoods across reactions.
    """
    num_reactions, K, _ = x_groups.shape
    diff = x_groups.unsqueeze(2) - x_groups.unsqueeze(1)  # (R, K, K, D)
    dist_sq = (diff ** 2).sum(dim=-1)  # (R, K, K)

    # Median heuristic bandwidth
    triu_idx = torch.triu_indices(K, K, offset=1)
    upper_dists = dist_sq[:, triu_idx[0], triu_idx[1]]
    median_dist = upper_dists.median(dim=-1).values.clamp(min=eps)
    bandwidth = 1.0 / median_dist.unsqueeze(-1).unsqueeze(-1)

    L = torch.exp(-bandwidth * dist_sq)
    I = torch.eye(K, device=L.device, dtype=L.dtype).unsqueeze(0)
    _, log_det_L = torch.linalg.slogdet(L + eps * I)
    _, log_det_LpI = torch.linalg.slogdet(L + I)
    return (log_det_L - log_det_LpI).sum()


def _zero_center_batch(x_batch, masks):
    """Zero-center each matrix in the batch to preserve electron conservation."""
    N = masks.view(x_batch.shape[0], -1).sum(dim=-1, keepdim=True).clamp(min=1)
    mean = (x_batch * masks).view(x_batch.shape[0], -1).sum(dim=-1, keepdim=True) / N
    return (x_batch - mean.view(-1, 1, 1) * masks) * masks


def diverse_ode_integrate(
    model, y_emb, y_len, x0_be, x0_cv,
    matrix_masks, node_masks,
    K, num_steps=20, gamma=5.0,
    diverse_be=True, diverse_cv=False,
):
    """
    Integrate the flow ODE from t=0 to t=1 with DPP diversity coupling,
    using torchdiffeq (Euler) and autograd for the DPP gradient.
    """
    use_chirality = getattr(model, 'use_chirality', True)
    B, n, _ = x0_be.shape
    num_reactions = B // K
    float_mm = matrix_masks.float()
    float_nm = node_masks.float()

    def velocity(t, state):
        x_be = state[0] if use_chirality else state
        x_cv = state[1] if use_chirality else None

        v_be, v_cv = model.forward(y_emb, y_len, x_be, t, x_cv)

        t_val = t.item()
        remaining = max(1.0 - t_val, 1e-6)

        if t_val >= 0.05 and t_val <= 0.95 and K > 1:
            if diverse_be:
                with torch.enable_grad():
                    x_in = x_be.detach().requires_grad_(True)
                    x_hat = (x_in + v_be.detach() * remaining) * float_mm
                    log_L = _dpp_log_likelihood(x_hat.view(B, -1).view(num_reactions, K, -1))
                    grad_be = torch.autograd.grad(log_L, x_in)[0]

                # Symmetrize and zero-center (BE matrix chemical constraints)
                grad_be = 0.5 * (grad_be + grad_be.transpose(1, 2))
                grad_be = _zero_center_batch(grad_be, float_mm)
                grad_be = grad_be * float_mm
                norm = grad_be.view(B, -1).norm(dim=-1).clamp(min=1e-8).view(B, 1, 1)
                v_be = v_be - gamma * (1.0 - t_val) / norm * grad_be

            if diverse_cv and use_chirality:
                with torch.enable_grad():
                    cv_in = x_cv.detach().requires_grad_(True)
                    cv_hat = (cv_in + v_cv.detach() * remaining) * float_nm
                    log_L_cv = _dpp_log_likelihood(cv_hat.view(B, -1).view(num_reactions, K, -1))
                    grad_cv = torch.autograd.grad(log_L_cv, cv_in)[0]

                grad_cv = grad_cv * float_nm
                norm_cv = grad_cv.view(B, -1).norm(dim=-1).clamp(min=1e-8).view(B, 1)
                v_cv = v_cv - gamma * (1.0 - t_val) / norm_cv * grad_cv

        v_be = v_be.masked_fill(~matrix_masks.bool(), 0)
        if use_chirality:
            v_cv = v_cv.masked_fill(~node_masks.bool(), 0)
            return (v_be, v_cv)
        return v_be

    t_span = torch.linspace(0, 1, num_steps + 1, device=x0_be.device)
    init = (x0_be, x0_cv) if use_chirality else x0_be
    result = torchdiffeq.odeint(velocity, init, t_span, method="euler")

    if use_chirality:
        traj_be, traj_cv = result
        return torch.stack([traj_be[0], traj_be[-1]]), torch.stack([traj_cv[0], traj_cv[-1]])
    else:
        return torch.stack([result[0], result[-1]]), None
