"""
ImitationNet reproduction — networks (paper §III-C, Fig.2).

  Q_h: human pose (4 limb dirs, 12-d) -> shared latent z (d=8)
  Q_r: robot pose (14 arm joint angles) -> shared latent z (d=8)
  D_r: latent z (d=8) -> robot pose (14 arm joint angles)

All are MLPs with 6 hidden layers of 128 units (paper §IV-A). Latent dim = 8.
"""
import torch
import torch.nn as nn

HUMAN_DIM = 4 * 6   # 4 limb joints, local 6D rotation each = 24 (paper: joint quaternions)
ROBOT_DIM = 14      # 14 arm joint angles
LATENT = 8
HIDDEN = 128
N_LAYERS = 6


def _mlp(in_dim, out_dim, hidden=HIDDEN, layers=N_LAYERS):
    mods = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*mods)


class ImitationNet(nn.Module):
    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.latent = latent
        self.Qh = _mlp(HUMAN_DIM, latent, hidden)   # human encoder
        self.Qr = _mlp(ROBOT_DIM, latent, hidden)   # robot encoder
        self.Dr = _mlp(latent, ROBOT_DIM, hidden)   # robot decoder

    def enc_h(self, xh):   # xh: (B,4,3) or (B,12)
        return self.Qh(xh.reshape(xh.shape[0], -1))

    def enc_r(self, xr):   # xr: (B,14)
        return self.Qr(xr)

    def dec_r(self, z):    # z: (B,8) -> (B,14), bounded to normalized joint range [-1,1]
        return torch.tanh(self.Dr(z))

    @torch.no_grad()
    def retarget(self, xh):
        """Inference: human pose (limb dirs) -> robot joint angles."""
        return self.dec_r(self.enc_h(xh))


if __name__ == "__main__":
    m = ImitationNet()
    print("params:", sum(p.numel() for p in m.parameters()))
    xh = torch.randn(4, 4, 3); xr = torch.randn(4, 14)
    print("enc_h:", tuple(m.enc_h(xh).shape), "enc_r:", tuple(m.enc_r(xr).shape),
          "dec_r:", tuple(m.dec_r(m.enc_h(xh)).shape))
