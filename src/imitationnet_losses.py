"""
ImitationNet reproduction — losses (paper Eq. 2-5).

  L_triplet (Eq2): max(||z_a - z_pos|| - ||z_a - z_neg|| + alpha, 0), alpha=0.05
      anchor z_a = human pose; positive = robot pose similar to anchor (low S_RD);
      negative = human pose dissimilar to anchor (high S_RD).   [Fig.2]
  L_rec (Eq3): ||x_r - D_r(Q_r(x_r))||_1                 robot autoencoder
  L_ltc (Eq4): ||Q_h(x_h) - Q_r(D_r(Q_h(x_h)))||_1       human->latent->robot->latent cycle
  total (Eq5): lambda_t * L_triplet + lambda_r * L_rec + L_ltc, lambda_t=10, lambda_r=5
"""
import torch
import torch.nn.functional as F

ALPHA = 0.05
LAMBDA_T = 10.0
LAMBDA_R = 5.0   # paper Eq.5: lambda_rec = 5 (was wrongly 1.0)


def triplet_loss(z_a, z_pos, z_neg, alpha=ALPHA):
    d_pos = (z_a - z_pos).norm(dim=-1)
    d_neg = (z_a - z_neg).norm(dim=-1)
    return F.relu(d_pos - d_neg + alpha).mean()


def imitation_loss(model, xh_a, xr_pos, xh_neg, xr_recon):
    """
    xh_a   : (B,4,3) human anchor limb dirs
    xr_pos : (B,14)  robot pose selected as positive (similar to anchor)
    xh_neg : (B,4,3) human limb dirs selected as negative (dissimilar to anchor)
    xr_recon:(B,14)  robot poses for the reconstruction loss (any sampled robot poses)
    """
    z_a = model.enc_h(xh_a)
    z_pos = model.enc_r(xr_pos)
    z_neg = model.enc_h(xh_neg)
    l_tri = triplet_loss(z_a, z_pos, z_neg)

    # reconstruction (robot autoencoder)
    xr_hat = model.dec_r(model.enc_r(xr_recon))
    l_rec = (xr_recon - xr_hat).abs().mean()

    # latent consistency: human -> latent -> robot -> latent
    z_h = model.enc_h(xh_a)
    z_cycle = model.enc_r(model.dec_r(z_h))
    l_ltc = (z_h - z_cycle).abs().mean()

    total = LAMBDA_T * l_tri + LAMBDA_R * l_rec + l_ltc
    logs = {"total": total.item(), "triplet": l_tri.item(),
            "rec": l_rec.item(), "ltc": l_ltc.item()}
    return total, logs
