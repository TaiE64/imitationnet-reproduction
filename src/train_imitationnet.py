"""
ImitationNet reproduction — training (unsupervised human->robot retargeting).

Builds a robot-pose bank (sampled joint angles + FK limb dirs) and a human-pose bank
(InterHuman limb dirs). Each step samples triplets:
  anchor   = human pose
  positive = robot pose most similar to anchor (min S_RD over C candidates)
  negative = human pose most dissimilar to anchor (max S_RD over C candidates)
and optimizes L_triplet + L_rec + L_ltc (paper Eq.2-5).
Adam, lr 1e-3, batch 256 (paper §IV-A).
"""
import argparse
import os
import torch
from robot_kinematics import RobotFK
from human_repr import load_humanml3d_poses, human_limb_frames_canon
from rotations import sixd_to_rotmat, rotmat_to_6d, s_rd_rot_pairwise, s_rd_rot
from imitationnet_model import ImitationNet
from imitationnet_losses import imitation_loss
from paths import CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_robot", type=int, default=50000)
    ap.add_argument("--cand", type=int, default=256)   # candidates for neg mining
    ap.add_argument("--pos_pool", type=int, default=300000)  # robot pool for CONSISTENT best-match positive
    ap.add_argument("--hstride", type=int, default=5)        # HumanML3D frame stride (1 -> ~2.8M poses, paper-scale)
    ap.add_argument("--lambda_fk", type=float, default=0.0)  # >0: differentiable-FK direct retargeting loss
    ap.add_argument("--pure_fk", action="store_true")        # train ONLY the FK loss (no triplet/rec/ltc)
    ap.add_argument("--latent", type=int, default=8)         # shared latent dim (paper=8)
    ap.add_argument("--hidden", type=int, default=128)       # MLP hidden width (paper=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(CKPT, "run"))   # output dir for this run
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.n_robot, args.cand, args.device, args.pos_pool = 200, 2000, 64, "cpu", 3000
        args.ckpt = os.path.join(CKPT, "smoke")

    os.makedirs(args.ckpt, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ---- robot bank: cached (paper-scale 15M) if present, else sample on the fly ----
    fk = RobotFK(device=dev)          # on training device (differentiable FK loss needs it)
    import numpy as np
    bank_path = "/home/taie/Downloads/Reproduce/datasets/robot_retargeting/robot_bank.npz"
    if os.path.exists(bank_path) and not args.smoke:
        z = np.load(bank_path)
        rb_pose = torch.from_numpy(z["poses"]).float()    # (N,14) normalized [-1,1]
        rb_rot6 = torch.from_numpy(z["rots"]).float()     # (N,24) limb global rot, 6D
        print(f"[robot bank] loaded {rb_pose.shape[0]} cached poses from {bank_path}")
    else:
        rb_pose_real = fk.sample_arm_poses(args.n_robot)
        rb_rot6 = rotmat_to_6d(fk.limb_frames_canon(rb_pose_real)).reshape(args.n_robot, -1)
        rb_pose = fk.normalize(rb_pose_real)
        print(f"[robot bank] sampled {args.n_robot} poses")
    rb_pose = rb_pose.to(dev)                 # poses on GPU (needed during training)
    # rb_rot6 (1.44GB at 15M) stays on CPU -> only the pos-mining pool subset goes to GPU,
    # so the FULL HumanML3D (stride 1, all frames) fits on a 16GB GPU.

    # ---- human bank: body-canonical limb frames -> input (6D) + similarity (rotmats) ----
    J = load_humanml3d_poses("train", max_motions=50 if args.smoke else None, stride=args.hstride)
    h_rot = human_limb_frames_canon(J).to(dev)            # (M,4,3,3) canonical limb frames
    h_in = rotmat_to_6d(h_rot).reshape(h_rot.shape[0], -1) # (M,24) network input x_h (6D)
    print(f"[human bank] {h_in.shape[0]} poses (HumanML3D, body-canonical rotation S_RD)")

    # ---- precompute a CONSISTENT, tight positive robot for each human (paper: "the robot
    # pose most similar through the similarity metric"). Random-per-step positives make z_h
    # blurry; a fixed best-match over a large pool gives a stable target to align to. ----
    M0, Nrob = h_in.shape[0], rb_pose.shape[0]
    pool = min(args.pos_pool, Nrob)
    pidx = torch.randperm(Nrob)[:pool]                             # CPU indices
    Rr_pool = sixd_to_rotmat(rb_rot6[pidx].reshape(pool, 4, 6)).to(dev)  # (pool,4,3,3) only subset on GPU
    pidx_dev = pidx.to(dev)
    pos_pose = torch.empty(M0, rb_pose.shape[1], device=dev)
    chunk = 200
    for i in range(0, M0, chunk):
        S = s_rd_rot_pairwise(h_rot[i:i + chunk], Rr_pool)         # (c,pool)
        pos_pose[i:i + chunk] = rb_pose[pidx_dev[S.argmin(dim=1)]]
    print(f"[positives] precomputed best-match robot for {M0} humans over {pool} robot pool")
    # h_rot only needed for positive mining; free it (+ pool) so training holds just h_in,
    # pos_pose, rb_pose on GPU -> the FULL HumanML3D fits.
    del rb_rot6, Rr_pool, h_rot
    torch.cuda.empty_cache()

    model = ImitationNet(latent=args.latent, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("params:", sum(p.numel() for p in model.parameters()))

    M, N = h_in.shape[0], rb_pose.shape[0]
    for step in range(args.steps):
        model.train()
        a_idx = torch.randint(0, M, (args.batch,), device=dev)
        xh_a = h_in[a_idx]                                 # (B,24) network input

        with torch.no_grad():
            # positive: CONSISTENT precomputed best-match robot for this anchor (stable target)
            xr_pos = pos_pose[a_idx]                       # (B,14)
            # negative: a RANDOM other human (paper uses random triplets, Fig.2). A random
            # negative is much closer than "argmax most-dissimilar" -> a non-trivial margin
            # that actually pulls z_h tight to its positive.
            neg_idx = torch.randint(0, M, (args.batch,), device=dev)
            recon_idx = torch.randint(0, N, (args.batch,), device=dev)

        xh_neg = h_in[neg_idx]
        xr_recon = rb_pose[recon_idx]

        if args.pure_fk:
            loss = torch.zeros((), device=dev); logs = {}    # FK loss only (no shared-latent losses)
        else:
            loss, logs = imitation_loss(model, xh_a, xr_pos, xh_neg, xr_recon)

        if args.lambda_fk > 0:
            # DIRECT differentiable-FK retargeting loss: push the retargeted robot's canonical
            # limb frames to match the human's, straight through forward kinematics. Optimizes
            # the actual objective (limb-orientation match) instead of the indirect latent triplet.
            Rh_a = sixd_to_rotmat(xh_a.reshape(args.batch, 4, 6))          # human canonical frames
            xr_pred = fk.denormalize(model.dec_r(model.enc_h(xh_a)))       # (B,14) joint angles
            R_pred = fk.limb_frames_canon(xr_pred)                          # differentiable FK
            l_fk = s_rd_rot(Rh_a, R_pred).mean()
            loss = loss + args.lambda_fk * l_fk
            logs["fk"] = l_fk.item()

        opt.zero_grad(); loss.backward(); opt.step()

        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"step {step:6d} | " + " ".join(f"{k}:{v:.4f}" for k, v in logs.items()))

    torch.save({"model": model.state_dict(), "latent": args.latent, "hidden": args.hidden},
               f"{args.ckpt}/imitationnet.pt")
    print("DONE -> saved", f"{args.ckpt}/imitationnet.pt")


if __name__ == "__main__":
    main()
