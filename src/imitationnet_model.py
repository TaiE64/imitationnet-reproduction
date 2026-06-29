"""
ImitationNet reproduction — networks (paper §III-C, Fig.2).

  Q_h: human pose (4 limbs x 6D rotation, 24-d) -> shared latent z
  Q_r: robot pose (14 arm joint angles)         -> shared latent z
  D_r: latent z -> robot pose (14 arm joint angles)

All are MLPs with 6 hidden layers of 128 units (paper §IV-A). Latent dim is 8 in the
paper (default here); the shipped best model uses 32 (see EXPERIMENTS.md).

NOTE on the human input: the paper §III-A specifies each human joint as a LOCAL
quaternion relative to its parent (n=4). We instead feed body-canonical GLOBAL limb
rotation frames as 6D (4 limbs x 6 = 24-d). This is a deliberate divergence (it makes
the human/robot frames coordinate-comparable); see EXPERIMENTS.md.
"""
import torch
import torch.nn as nn

HUMAN_DIM = 4 * 6   # 4 limbs, 6D body-canonical rotation each = 24
ROBOT_DIM = 14      # 14 arm joint angles
LATENT = 8          # paper §IV-A (best shipped model overrides to 32)
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

    def enc_h(self, xh):   # xh: (B,4,6) or (B,24)
        return self.Qh(xh.reshape(xh.shape[0], -1))

    def enc_r(self, xr):   # xr: (B,14)
        return self.Qr(xr)

    def dec_r(self, z):    # z: (B,latent) -> (B,14), bounded to normalized joint range [-1,1]
        return torch.tanh(self.Dr(z))

    @torch.no_grad()
    def retarget(self, xh):
        """Inference: human pose (4-limb 6D rotations) -> robot joint angles."""
        return self.dec_r(self.enc_h(xh))


if __name__ == "__main__":
    m = ImitationNet()
    print("params:", sum(p.numel() for p in m.parameters()))
    xh = torch.randn(4, 4, 6); xr = torch.randn(4, 14)   # human = 4 limbs x 6D
    print("enc_h:", tuple(m.enc_h(xh).shape), "enc_r:", tuple(m.enc_r(xr).shape),
          "dec_r:", tuple(m.dec_r(m.enc_h(xh)).shape))
