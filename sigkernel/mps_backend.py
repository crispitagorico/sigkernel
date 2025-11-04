import torch

def _get_diagonal_indices(diagonal_idx, len_x, len_y, device):
    """Extract (i,j) index pairs for all grid points on specified anti-diagonal.

    Input:
        - diagonal_idx: int, diagonal index from 0 to 2*max(len_x,len_y)-2
        - len_x: int, first path dimension
        - len_y: int, second path dimension
        - device: torch device for index tensors
    Output:
        - i_indices: 1D tensor of i coordinates
        - j_indices: 1D tensor of j coordinates
    """
    max_i = min(diagonal_idx, len_x - 1)
    min_i = max(0, diagonal_idx - (len_y - 1))

    i_indices = torch.arange(min_i, max_i + 1, device=device, dtype=torch.long)
    j_indices = diagonal_idx - i_indices

    return i_indices, j_indices


def sigkernel_mps(M_inc, len_x, len_y, M_sol, _naive_solver=False):
    """Signature kernel PDE solver for MPS backend using vectorized anti-diagonal sweep.

    Input:
        - M_inc: torch tensor of shape (batch, len_x, len_y), PDE increments
        - len_x: int, first path length
        - len_y: int, second path length
        - M_sol: torch tensor of shape (batch, len_x+2, len_y+2), solution matrix (modified in-place)
        - _naive_solver: bool, if True use first-order solver, else higher-order Padé
    """
    n_anti_diagonals = len_x + len_y - 1
    device = M_inc.device

    for p in range(n_anti_diagonals):
        i_idx, j_idx = _get_diagonal_indices(p, len_x, len_y, device)

        inc = M_inc[:, i_idx, j_idx]

        k_01 = M_sol[:, i_idx, j_idx + 1]
        k_10 = M_sol[:, i_idx + 1, j_idx]
        k_00 = M_sol[:, i_idx, j_idx]

        if _naive_solver:
            M_sol[:, i_idx + 1, j_idx + 1] = (k_01 + k_10) * (1. + 0.5 * inc) - k_00
        else:
            M_sol[:, i_idx + 1, j_idx + 1] = (k_01 + k_10) * (1. + 0.5 * inc + (1./12) * inc**2) - k_00 * (1. - (1./12) * inc**2)


def sigkernel_Gram_mps(M_inc, len_x, len_y, M_sol, _naive_solver=False):
    """Signature kernel Gram matrix PDE solver for MPS backend using vectorized anti-diagonal sweep.

    Input:
        - M_inc: torch tensor of shape (batch_X, batch_Y, len_x, len_y), PDE increments
        - len_x: int, first path length
        - len_y: int, second path length
        - M_sol: torch tensor of shape (batch_X, batch_Y, len_x+2, len_y+2), solution matrix (modified in-place)
        - _naive_solver: bool, if True use first-order solver, else higher-order Padé
    """
    n_anti_diagonals = len_x + len_y - 1
    device = M_inc.device

    for p in range(n_anti_diagonals):
        i_idx, j_idx = _get_diagonal_indices(p, len_x, len_y, device)

        inc = M_inc[:, :, i_idx, j_idx]

        k_01 = M_sol[:, :, i_idx, j_idx + 1]
        k_10 = M_sol[:, :, i_idx + 1, j_idx]
        k_00 = M_sol[:, :, i_idx, j_idx]

        if _naive_solver:
            M_sol[:, :, i_idx + 1, j_idx + 1] = (k_01 + k_10) * (1. + 0.5 * inc) - k_00
        else:
            M_sol[:, :, i_idx + 1, j_idx + 1] = (k_01 + k_10) * (1. + 0.5 * inc + (1./12) * inc**2) - k_00 * (1. - (1./12) * inc**2)


def sigkernel_derivatives_Gram_mps(M_inc, M_inc_diff, M_inc_diffdiff, len_x, len_y, M_sol, M_sol_diff, M_sol_diffdiff):
    """Signature kernel derivatives PDE solver for MPS backend using vectorized anti-diagonal sweep.

    Computes kernel and its first/second directional derivatives simultaneously.

    Input:
        - M_inc: torch tensor of shape (batch_X, batch_Y, len_x, len_y), PDE increments
        - M_inc_diff: torch tensor of shape (batch_X, batch_Y, len_x, len_y), first derivative increments
        - M_inc_diffdiff: torch tensor of shape (batch_X, batch_Y, len_x, len_y), second derivative increments
        - len_x: int, first path length
        - len_y: int, second path length
        - M_sol: torch tensor of shape (batch_X, batch_Y, len_x+2, len_y+2), kernel solution (modified in-place)
        - M_sol_diff: torch tensor of shape (batch_X, batch_Y, len_x+2, len_y+2), first derivative solution (modified in-place)
        - M_sol_diffdiff: torch tensor of shape (batch_X, batch_Y, len_x+2, len_y+2), second derivative solution (modified in-place)
    """
    n_anti_diagonals = len_x + len_y - 1
    device = M_inc.device

    for p in range(n_anti_diagonals):
        i_idx, j_idx = _get_diagonal_indices(p, len_x, len_y, device)

        inc = M_inc[:, :, i_idx, j_idx]
        inc_diff = M_inc_diff[:, :, i_idx, j_idx]
        inc_diffdiff = M_inc_diffdiff[:, :, i_idx, j_idx]

        k_01 = M_sol[:, :, i_idx, j_idx + 1]
        k_10 = M_sol[:, :, i_idx + 1, j_idx]
        k_00 = M_sol[:, :, i_idx, j_idx]

        k_01_diff = M_sol_diff[:, :, i_idx, j_idx + 1]
        k_10_diff = M_sol_diff[:, :, i_idx + 1, j_idx]
        k_00_diff = M_sol_diff[:, :, i_idx, j_idx]

        k_01_diffdiff = M_sol_diffdiff[:, :, i_idx, j_idx + 1]
        k_10_diffdiff = M_sol_diffdiff[:, :, i_idx + 1, j_idx]
        k_00_diffdiff = M_sol_diffdiff[:, :, i_idx, j_idx]

        M_sol[:, :, i_idx + 1, j_idx + 1] = (k_01 + k_10) * (1. + 0.5 * inc + (1./12) * inc**2) - k_00 * (1. - (1./12) * inc**2)

        f1 = k_00 * inc_diff + k_00_diff * inc
        f2 = k_01 * inc_diff + k_01_diff * inc
        f3 = k_10 * inc_diff + k_10_diff * inc
        f4 = M_sol[:, :, i_idx + 1, j_idx + 1] * inc_diff + (k_01_diff + k_10_diff - k_00_diff + f1) * inc
        M_sol_diff[:, :, i_idx + 1, j_idx + 1] = k_01_diff + k_10_diff - k_00_diff + 0.25 * (f1 + f2 + f3 + f4)

        g1 = k_00 * inc_diffdiff + 2. * k_00_diff * inc_diff + k_00_diffdiff * inc
        g2 = k_01 * inc_diffdiff + 2. * k_01_diff * inc_diff + k_01_diffdiff * inc
        g3 = k_10 * inc_diffdiff + 2. * k_10_diff * inc_diff + k_10_diffdiff * inc
        g4 = M_sol[:, :, i_idx + 1, j_idx + 1] * inc_diffdiff + 2. * M_sol_diff[:, :, i_idx + 1, j_idx + 1] * inc_diff + (k_01_diffdiff + k_10_diffdiff - k_00_diffdiff + g1) * inc
        M_sol_diffdiff[:, :, i_idx + 1, j_idx + 1] = k_01_diffdiff + k_10_diffdiff - k_00_diffdiff + 0.25 * (g1 + g2 + g3 + g4)
