"""
Best-quality retargeting viz: pure-FK network init + test-time refinement (~5.3 deg).
Human (3-joint arm) vs retargeted TIAGo++ (full 7-DOF chain), body-canonicalized.
Renders a 6-pose static strip AND a refined animation.
"""
import os, glob, random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from robot_kinematics import RobotFK
from human_repr import load_humanml3d_poses, human_limb_frames_canon, HML3D_ROOT
from rotations import rotmat_to_6d, body_frame, s_rd_rot
from imitationnet_model import ImitationNet
from paths import ckpt, media

fk = RobotFK()
_c = torch.load(ckpt("purefk"), map_location="cpu")
model = ImitationNet(latent=_c.get("latent", 8), hidden=_c.get("hidden", 128))
model.load_state_dict(_c["model"]); model.eval()
LARM = [f"arm_left_{i}_link" for i in range(1, 8)]
RARM = [f"arm_right_{i}_link" for i in range(1, 8)]


def retarget_refined(Rh, steps=60):
    """network init + test-time FK refinement -> robot joint angles (radians)."""
    h_in = rotmat_to_6d(Rh).reshape(Rh.shape[0], -1)
    with torch.no_grad():
        pred = model.retarget(h_in)
    theta = torch.atanh(pred.clamp(-0.999, 0.999)).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.03)
    for _ in range(steps):
        R = fk.limb_frames_canon(fk.denormalize(torch.tanh(theta)))
        loss = s_rd_rot(Rh, R).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return fk.denormalize(torch.tanh(theta)).detach()


def canon(pts, lsh, rsh):
    B = body_frame(lsh, rsh); c = (lsh + rsh) / 2; w = (lsh - rsh).norm() + 1e-8
    return ((pts - c) @ B) / w


def setup(ax, title):
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_zlim(-2.1, 1.3)
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=12, azim=-70); ax.axis("off")
    ax.set_title(title, fontsize=9)


def draw_human(ax, hc):
    for a, b, c in [(16, 18, 20), (17, 19, 21)]:
        arm = torch.stack([hc[a], hc[b], hc[c]]).numpy()
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color="tab:blue", lw=3, ms=6)
    ax.plot(*torch.stack([hc[16], hc[17]]).numpy().T, "-", color="gray", lw=2)


def draw_robot(ax, rp, i):
    lsh, rsh = rp["arm_left_2_link"][i], rp["arm_right_2_link"][i]
    for chain in (LARM, RARM):
        pts = canon(torch.stack([rp[n][i] for n in chain]), lsh, rsh).numpy()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="tab:red", lw=2.5, ms=5)
    ax.plot(*canon(torch.stack([lsh, rsh]), lsh, rsh).numpy().T, "-", color="gray", lw=2)


# ---------- static 6-pose strip ----------
J = load_humanml3d_poses("test", max_motions=400, stride=15)
elev = (J[:, 20, 2] + J[:, 21, 2]) / 2 - (J[:, 16, 2] + J[:, 17, 2]) / 2
idx = torch.argsort(elev)[torch.linspace(0, len(elev) - 1, 6).long()]
Js = J[idx]; Rh = human_limb_frames_canon(Js)
xr = retarget_refined(Rh)
rp = {k: v.get_matrix()[:, :3, 3] for k, v in fk.chain.forward_kinematics(fk._full_config(xr)).items()}
fig = plt.figure(figsize=(18, 6))
for i in range(6):
    hc = canon(Js[i], Js[i][16], Js[i][17])
    a1 = fig.add_subplot(2, 6, i + 1, projection="3d"); setup(a1, "HUMAN" if i == 0 else ""); draw_human(a1, hc)
    a2 = fig.add_subplot(2, 6, i + 7, projection="3d"); setup(a2, "ROBOT (refined ~5deg)" if i == 0 else ""); draw_robot(a2, rp, i)
plt.tight_layout(); plt.savefig(media("retarget_best.png"), dpi=95, bbox_inches="tight")
print("saved retarget_best.png")

# ---------- refined animation ----------
files = glob.glob(os.path.join(HML3D_ROOT, "new_joints", "*.npy")); random.shuffle(files)
for f in files:
    m = np.load(f)
    if m.ndim == 3 and m.shape[1] == 22 and 80 <= m.shape[0] <= 200 and \
       float(np.var(m[:, 20], 0).sum() + np.var(m[:, 21], 0).sum()) > 1.5:
        break
print("motion:", os.path.basename(f))
Ja = torch.from_numpy(np.load(f)).float()[::2]; Ta = Ja.shape[0]
Rha = human_limb_frames_canon(Ja)
xra = retarget_refined(Rha)
rpa = {k: v.get_matrix()[:, :3, 3] for k, v in fk.chain.forward_kinematics(fk._full_config(xra)).items()}
Hc = [canon(Ja[t], Ja[t][16], Ja[t][17]) for t in range(Ta)]
figA = plt.figure(figsize=(10, 5)); ax1 = figA.add_subplot(1, 2, 1, projection="3d"); ax2 = figA.add_subplot(1, 2, 2, projection="3d")


def upd(t):
    ax1.cla(); ax2.cla(); setup(ax1, "HUMAN"); setup(ax2, "ROBOT (refined ~5deg)")
    draw_human(ax1, Hc[t]); draw_robot(ax2, rpa, t)
    figA.suptitle(f"frame {t+1}/{Ta}", fontsize=9)


FuncAnimation(figA, upd, frames=Ta, interval=60).save(
    media("retarget_best.gif"), writer=PillowWriter(fps=15))
print("saved retarget_best.gif")
