"""
Animate ImitationNet retargeting over a continuous human motion: human arms (left) vs
retargeted TIAGo++ arms (right), both in the body frame. Saves a GIF.
"""
import os, glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from robot_kinematics import RobotFK
from human_repr import human_limb_frames_canon, HML3D_ROOT
from rotations import rotmat_to_6d, body_frame
from imitationnet_model import ImitationNet
from paths import ckpt, media

fk = RobotFK()
_c = torch.load(ckpt("best"), map_location="cpu")
model = ImitationNet(latent=_c.get("latent", 8), hidden=_c.get("hidden", 128))
model.load_state_dict(_c["model"])
model.eval()

# RANDOMLY pick a motion with visible arm movement (length 80-250, wrist variance > 1.5)
import random
jdir = os.path.join(HML3D_ROOT, "new_joints")
files = glob.glob(os.path.join(jdir, "*.npy"))
random.shuffle(files)
best = None
for f in files:
    m = np.load(f)
    if m.ndim != 3 or m.shape[1] != 22 or not (80 <= m.shape[0] <= 250):
        continue
    v = float(np.var(m[:, 20], 0).sum() + np.var(m[:, 21], 0).sum())
    if v > 1.5:                       # enough motion to be interesting, else keep looking
        best, bestvar = f, v
        break
print("motion:", os.path.basename(best), "var", round(bestvar, 3))
J = torch.from_numpy(np.load(best)).float()                   # (T,22,3)
J = J[::2]                                                     # ~ halve fps for smaller gif
T = J.shape[0]

Rh = human_limb_frames_canon(J)
h_in = rotmat_to_6d(Rh).reshape(T, -1)
with torch.no_grad():
    xr = fk.denormalize(model.retarget(h_in))                 # (T,14)
full = fk._full_config(xr)
rpos = {k: v.get_matrix()[:, :3, 3] for k, v in fk.chain.forward_kinematics(full).items()}


def canon(pts, lsh, rsh):
    B = body_frame(lsh, rsh); c = (lsh + rsh) / 2; w = (lsh - rsh).norm() + 1e-8
    return ((pts - c) @ B) / w


LARM = [f"arm_left_{i}_link" for i in range(1, 8)]    # full 7-link robot chain
RARM = [f"arm_right_{i}_link" for i in range(1, 8)]

# precompute canonicalized arm points per frame
H, Rb = [], []
for t in range(T):
    j = J[t]
    hc = canon(j, j[16], j[17])
    H.append((torch.stack([hc[16], hc[18], hc[20]]), torch.stack([hc[17], hc[19], hc[21]]),
              torch.stack([hc[16], hc[17]])))
    lsh, rsh = rpos["arm_left_2_link"][t], rpos["arm_right_2_link"][t]
    lc = canon(torch.stack([rpos[n][t] for n in LARM]), lsh, rsh)
    rc = canon(torch.stack([rpos[n][t] for n in RARM]), lsh, rsh)
    Rb.append((lc, rc, canon(torch.stack([lsh, rsh]), lsh, rsh)))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")


def setup(ax, title):
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_zlim(-2.1, 1.3)
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=12, azim=-70); ax.axis("off")
    ax.set_title(title, fontsize=11)


def draw(ax, P, color, ms):
    larm, rarm, sh = P
    for arm in (larm.numpy(), rarm.numpy()):
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color=color, lw=2.5, ms=ms)
    ax.plot(sh.numpy()[:, 0], sh.numpy()[:, 1], sh.numpy()[:, 2], "-", color="gray", lw=2)


def update(t):
    ax1.cla(); ax2.cla()
    setup(ax1, "HUMAN (3 joints/arm)"); setup(ax2, "ROBOT (7 joints/arm)")
    draw(ax1, H[t], "tab:blue", 6); draw(ax2, Rb[t], "tab:red", 5)
    fig.suptitle(f"frame {t+1}/{T}", fontsize=9)


anim = FuncAnimation(fig, update, frames=T, interval=60)
out = media("retarget_anim.gif")
anim.save(out, writer=PillowWriter(fps=15))
print("saved", out)
