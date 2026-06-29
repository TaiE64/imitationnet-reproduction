"""
ImitationNet reproduction — cross-domain similarity metric S_RD (paper Eq. 1).

Paper: S_RD(x_h, x_r) = sum_limb (1 - <q_h, q_r>^2), comparing the GLOBAL ROTATION
of the body limbs between a human and a robot pose.

A human skeleton (positions only) gives a limb DIRECTION but no twist, so we compare
limb directions: for unit directions, (1 - <d_h, d_r>^2) = sin^2(theta) where theta is
the angle between the two limb directions — exactly the paper's formula structure,
specialized to the (twist-free) information available from a skeleton. Lower = more similar.
"""
import torch

N_LIMB = 4


def s_rd_pairwise(dirs_h, dirs_r):
    """
    dirs_h: (H,4,3), dirs_r: (R,4,3)  unit limb directions.
    Returns (H,R) similarity-distance matrix S_RD (lower = more similar).
    """
    # cos angle per limb between every human-robot pair: (H,R,4)
    cos = torch.einsum("hld,rld->hrl", dirs_h, dirs_r)
    s = (1.0 - cos ** 2).sum(dim=-1)      # sum over 4 limbs -> (H,R)
    return s


def s_rd(dirs_h, dirs_r):
    """Matched pairs: dirs_h,(B,4,3) dirs_r,(B,4,3) -> (B,) S_RD per pair."""
    cos = (dirs_h * dirs_r).sum(dim=-1)   # (B,4)
    return (1.0 - cos ** 2).sum(dim=-1)   # (B,)


if __name__ == "__main__":
    h = torch.nn.functional.normalize(torch.randn(3, 4, 3), dim=-1)
    r = torch.nn.functional.normalize(torch.randn(5, 4, 3), dim=-1)
    M = s_rd_pairwise(h, r)
    print("pairwise S_RD:", tuple(M.shape), "range", round(M.min().item(), 3), "-", round(M.max().item(), 3))
    print("self-similarity (should be ~0):", round(s_rd(h, h).max().item(), 5))
