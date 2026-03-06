"""
DiverseFlow: DPP-based diversity-promoting sampling for flow matching.

Implements the method from "DiverseFlow: Sample-Efficient Diverse Mode Coverage
in Flows" (Morshed & Boddeti, 2025, arXiv:2504.07894).

Instead of independent ODE trajectories, K samples per reaction are jointly
evolved with a DPP repulsive term that promotes diversity among predicted
products while preserving chemical constraints (electron conservation,
BE matrix symmetry).
"""

import torch
from model.flow_matching import zero_center_func


def compute_dpp_gradient(x_hat_1_groups, K, h=1.0, eps=1e-6):
    """
    Compute the DPP log-likelihood gradient for diversity among K target estimates.

    Given K estimated targets x_hat_1 (one per sample of the same reaction),
    constructs an RBF kernel L, computes grad log det(L) - grad log det(L + I),
    and returns the gradient w.r.t. each x_hat_1.

    Parameters
    ----------
    x_hat_1_groups : Tensor, shape (num_reactions, K, D)
        Flattened estimated targets grouped by reaction.
    K : int
        Number of samples per reaction.
    h : float
        RBF kernel bandwidth scaling factor.
    eps : float
        Numerical stability floor.

    Returns
    -------
    grad : Tensor, shape (num_reactions, K, D)
        Gradient of DPP log-likelihood w.r.t. x_hat_1.
    """
    num_reactions, K_actual, D = x_hat_1_groups.shape

    # Pairwise squared distances: (num_reactions, K, K)
    diff = x_hat_1_groups.unsqueeze(2) - x_hat_1_groups.unsqueeze(1)  # (R, K, K, D)
    dist_sq = (diff ** 2).sum(dim=-1)  # (R, K, K)

    # Median heuristic for bandwidth per reaction
    triu_indices = torch.triu_indices(K_actual, K_actual, offset=1)
    upper_dists = dist_sq[:, triu_indices[0], triu_indices[1]]  # (R, K*(K-1)/2)
    median_dist = upper_dists.median(dim=-1).values.clamp(min=eps)  # (R,)
    bandwidth = h / median_dist.unsqueeze(-1).unsqueeze(-1)  # (R, 1, 1)

    # RBF kernel: L_ij = exp(-bandwidth * ||x_i - x_j||^2)
    L = torch.exp(-bandwidth * dist_sq)  # (R, K, K)

    # DPP log-likelihood: log det(L) - log det(L + I)
    # Gradient: L_inv @ dL/dx - (L+I)_inv @ dL/dx
    # where dL_ij/dx_i = -2 * bandwidth * (x_i - x_j) * L_ij

    I = torch.eye(K_actual, device=L.device).unsqueeze(0).expand_as(L)
    L_plus_I = L + I

    # Stable inverse via Cholesky
    try:
        L_inv = torch.linalg.inv(L + eps * I)
        LpI_inv = torch.linalg.inv(L_plus_I)
    except torch.linalg.LinAlgError:
        L_inv = torch.linalg.pinv(L + eps * I)
        LpI_inv = torch.linalg.pinv(L_plus_I)

    # C = L_inv - (L+I)_inv, shape (R, K, K)
    C = L_inv - LpI_inv

    # Gradient for each sample i:
    # grad_i = sum_j  C_ij * L_ij * (-2 * bandwidth) * (x_i - x_j)
    # C_L = C * L, shape (R, K, K)
    C_L = C * L  # (R, K, K)

    # grad = -2 * bandwidth * sum_j C_L_ij * (x_i - x_j)
    # diff shape: (R, K, K, D), C_L shape: (R, K, K)
    grad = -2.0 * bandwidth.unsqueeze(-1) * (C_L.unsqueeze(-1) * diff).sum(dim=2)  # (R, K, D)

    return grad


def diverse_euler_step(
    x_be, x_cv, v_be, v_cv, t, dt, K,
    matrix_masks, node_masks,
    gamma=5.0, diverse_be=True, diverse_cv=False,
):
    """
    Perform one Euler step with DPP diversity coupling.

    Parameters
    ----------
    x_be : Tensor, shape (B, n, n)
        Current BE matrix states (B = num_reactions * K).
    x_cv : Tensor, shape (B, n)
        Current chiral vector states.
    v_be : Tensor, shape (B, n, n)
        Predicted BE velocity field.
    v_cv : Tensor, shape (B, n)
        Predicted chiral velocity field.
    t : float
        Current time.
    dt : float
        Time step size.
    K : int
        Number of coupled samples per reaction.
    matrix_masks : Tensor, shape (B, n, n)
        BE matrix masks (True = valid).
    node_masks : Tensor, shape (B, n)
        Node masks (True = valid).
    gamma : float
        Diversity strength scaling factor.
    diverse_be : bool
        Apply diversity to BE matrices.
    diverse_cv : bool
        Apply diversity to chiral vectors.

    Returns
    -------
    x_be_new : Tensor, shape (B, n, n)
        Updated BE matrix states.
    x_cv_new : Tensor, shape (B, n)
        Updated chiral vector states.
    """
    B, n, _ = x_be.shape
    num_reactions = B // K
    remaining_time = max(1.0 - t, 1e-6)

    if t < 0.05 or t > 0.95 or K <= 1:
        # No diversity at extreme timesteps or single sample
        x_be_new = x_be + v_be * dt
        x_cv_new = x_cv + v_cv * dt
        return x_be_new, x_cv_new

    # Estimate targets at t=1
    x_hat_be = x_be + v_be * remaining_time  # (B, n, n)
    x_hat_cv = x_cv + v_cv * remaining_time  # (B, n)

    # Apply diversity to BE matrices
    if diverse_be:
        # Flatten and group by reaction
        x_hat_flat = (x_hat_be * matrix_masks).view(B, -1)  # (B, n*n)
        x_hat_groups = x_hat_flat.view(num_reactions, K, -1)  # (R, K, n*n)

        dpp_grad_flat = compute_dpp_gradient(x_hat_groups, K)  # (R, K, n*n)
        dpp_grad_be = dpp_grad_flat.view(B, n, n)

        # Symmetrize the gradient (BE matrices are symmetric)
        dpp_grad_be = 0.5 * (dpp_grad_be + dpp_grad_be.transpose(1, 2))

        # Zero-center to preserve electron conservation
        dpp_grad_be = _zero_center_batch(dpp_grad_be, matrix_masks)

        # Mask padding
        dpp_grad_be = dpp_grad_be * matrix_masks

        # Adaptive scaling: normalize by gradient magnitude
        grad_norm = dpp_grad_be.view(B, -1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sigma_t = 1.0 - t  # proxy for noise level at time t
        scale = gamma * sigma_t / grad_norm.view(B, 1, 1)
        dpp_grad_be = scale * dpp_grad_be

        v_be_diverse = v_be - dpp_grad_be
    else:
        v_be_diverse = v_be

    if diverse_cv:
        x_hat_cv_groups = (x_hat_cv * node_masks).view(num_reactions, K, -1)
        dpp_grad_cv_flat = compute_dpp_gradient(x_hat_cv_groups, K)
        dpp_grad_cv = dpp_grad_cv_flat.view(B, n)
        dpp_grad_cv = dpp_grad_cv * node_masks
        grad_norm_cv = dpp_grad_cv.view(B, -1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sigma_t = 1.0 - t
        scale_cv = gamma * sigma_t / grad_norm_cv.view(B, 1)
        v_cv_diverse = v_cv - scale_cv * dpp_grad_cv
    else:
        v_cv_diverse = v_cv

    x_be_new = x_be + v_be_diverse * dt
    x_cv_new = x_cv + v_cv_diverse * dt

    return x_be_new, x_cv_new


def _zero_center_batch(x_batch, masks):
    """Zero-center each matrix in the batch to preserve electron conservation."""
    float_masks = masks.float()
    N = float_masks.view(x_batch.shape[0], -1).sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
    mean = (x_batch * float_masks).view(x_batch.shape[0], -1).sum(dim=-1, keepdim=True) / N  # (B, 1)
    x_centered = x_batch - mean.view(-1, 1, 1) * float_masks
    return x_centered * float_masks


def diverse_ode_integrate(
    model, y_emb, y_len, x0_be, x0_cv,
    matrix_masks, node_masks,
    K, num_steps=20, gamma=5.0,
    diverse_be=True, diverse_cv=False,
):
    """
    Integrate the flow ODE from t=0 to t=1 using Euler steps with DPP
    diversity coupling between K samples of the same reaction.

    Parameters
    ----------
    model : AttnEncoderXL
        The trained flow model.
    y_emb : Tensor, shape (B, n, d)
        Atom embeddings (already repeated K times per reaction).
    y_len : Tensor, shape (B,)
        Sequence lengths.
    x0_be : Tensor, shape (B, n, n)
        Initial BE matrix states (noised reactants).
    x0_cv : Tensor, shape (B, n)
        Initial chiral vector states.
    matrix_masks : Tensor, shape (B, n, n)
        BE matrix masks.
    node_masks : Tensor, shape (B, n)
        Node masks.
    K : int
        Number of coupled samples per reaction.
    num_steps : int
        Number of Euler integration steps.
    gamma : float
        Diversity strength.
    diverse_be : bool
        Apply diversity coupling to BE matrices.
    diverse_cv : bool
        Apply diversity coupling to chiral vectors.

    Returns
    -------
    traj_be : Tensor, shape (2, B, n, n)
        BE trajectory (initial and final states, matching torchdiffeq format).
    traj_cv : Tensor, shape (2, B, n)
        Chiral trajectory (initial and final states).
    """
    use_chirality = getattr(model, 'use_chirality', True)

    dt = 1.0 / num_steps
    x_be = x0_be.clone()
    x_cv = x0_cv.clone() if use_chirality else None

    # Store initial state
    init_be = x0_be.clone()
    init_cv = x0_cv.clone() if use_chirality else None

    float_matrix_masks = matrix_masks.float()
    float_node_masks = node_masks.float()

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.tensor(t, device=x_be.device, dtype=x_be.dtype)

        # Get velocity from model
        v_be, v_cv = model.forward(y_emb, y_len, x_be, t_tensor, x_cv)

        # Euler step with diversity
        if use_chirality:
            x_be, x_cv = diverse_euler_step(
                x_be, x_cv, v_be, v_cv, t, dt, K,
                float_matrix_masks, float_node_masks,
                gamma=gamma, diverse_be=diverse_be, diverse_cv=diverse_cv,
            )
            x_cv = x_cv.masked_fill(~node_masks.bool(), 0)
        else:
            # BE-only: pass zeros for cv, ignore cv output
            dummy_cv = torch.zeros(x_be.shape[0], x_be.shape[1], device=x_be.device)
            dummy_v_cv = torch.zeros_like(dummy_cv)
            x_be, _ = diverse_euler_step(
                x_be, dummy_cv, v_be, dummy_v_cv, t, dt, K,
                float_matrix_masks, float_node_masks,
                gamma=gamma, diverse_be=diverse_be, diverse_cv=False,
            )

        # Re-mask padding
        x_be = x_be.masked_fill(~matrix_masks.bool(), 0)

    # Return in torchdiffeq format: (time_steps, batch, ...)
    traj_be = torch.stack([init_be, x_be], dim=0)
    traj_cv = torch.stack([init_cv, x_cv], dim=0) if use_chirality else None

    return traj_be, traj_cv
