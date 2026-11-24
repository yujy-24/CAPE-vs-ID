"""Environment / data / backbone sanity check. Run first; everything must be green.

usage (WSL): ~/miniconda3/envs/lawam/bin/python /mnt/e/world_model/id_vs_cape/00_check_env.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from common import BB_DIR, DATA_ROOT, IMG

OK, BAD = "[OK]", "[!!]"


def check(name, cond, detail=""):
    print(f"{OK if cond else BAD} {name} {detail}")
    return cond


def main():
    good = True
    good &= check("torch cuda", torch.cuda.is_available(),
                  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    print(f"    VRAM {vram:.1f} GB")

    # ---- dataset layout ----
    good &= check("data root", os.path.isdir(DATA_ROOT))
    eps = sorted(glob.glob(os.path.join(DATA_ROOT, "meta/episodes/chunk-*/*.parquet")))
    dat = sorted(glob.glob(os.path.join(DATA_ROOT, "data/chunk-*/*.parquet")))
    vids = sorted(glob.glob(os.path.join(DATA_ROOT, "videos/observation.images.image/chunk-*/*.mp4")))
    good &= check("episodes parquets", len(eps) > 0, f"x{len(eps)}")
    good &= check("data parquets", len(dat) > 0, f"x{len(dat)}")
    good &= check("video files", len(vids) > 0, f"x{len(vids)}")

    ep_df = pd.concat([pd.read_parquet(p) for p in eps], ignore_index=True)
    info = json.load(open(os.path.join(DATA_ROOT, "meta/info.json")))
    good &= check("episode count", len(ep_df) == info["total_episodes"],
                  f"{len(ep_df)} vs {info['total_episodes']}")
    good &= check("fps", info["fps"] == 20)

    # per-episode video segment vs length consistency (timestamp span * fps ~ length)
    spans = (ep_df["videos/observation.images.image/to_timestamp"]
             - ep_df["videos/observation.images.image/from_timestamp"]).values
    est = np.round(spans * info["fps"]).astype(int)
    diff = est - ep_df["length"].values
    good &= check("ts span ~ length", np.abs(diff).max() <= 1,
                  f"max|diff|={np.abs(diff).max()}")

    # actions
    d0 = pd.read_parquet(dat[0], columns=["episode_index", "frame_index", "action", "task_index"])
    good &= check("action dim", d0["action"].iloc[0].shape == (7,))
    n_tasks = d0["task_index"].nunique()
    print(f"    tasks in first file: {n_tasks}; total_frames={info['total_frames']}")

    # ---- video decode ----
    # dataset videos are AV1 (fourcc AV01): cv2 (pip) has no AV1 decoder, and
    # torchvision whole-file reads OOM (up to ~100k frames per file -> ~20 GB).
    # PyAV's bundled libdav1d is the decoder build_cache streams with.
    row0 = ep_df.iloc[0]
    vpath = os.path.join(DATA_ROOT, "videos/observation.images.image",
                         f"chunk-{int(row0['videos/observation.images.image/chunk_index']):03d}",
                         f"file-{int(row0['videos/observation.images.image/file_index']):03d}.mp4")
    good &= check("ep0 video exists", os.path.isfile(vpath), vpath.split("videos/")[-1])
    backend = None
    try:
        import av
        with av.open(vpath) as c:
            st = c.streams.video[0]
            codec, fps, n_total = st.codec_context.name, float(st.average_rate), st.frames
            it = c.decode(video=0)
            frames = [next(it).to_ndarray(format="rgb24") for _ in range(60)]
        if len(frames) == 60 and frames[0].shape == (256, 256, 3):
            backend = "pyav-sequential"
            print(f"{OK} PyAV sequential decode: 60 frames {frames[0].shape}, "
                  f"codec={codec}, fps={fps:.2f}, container frames={n_total}")
            good &= check("container fps ~20", abs(fps - 20) < 0.5, f"{fps:.3f}")
        else:
            print(f"{BAD} PyAV decode failed (n={len(frames)}, {frames[0].shape if frames else None})")
    except ImportError:
        print(f"{BAD} PyAV not installed - build_cache cannot decode AV1")
    print(f"    -> build_cache will use backend: {backend}")

    # ---- backbone ----
    good &= check("backbone dir", os.path.isdir(BB_DIR))
    from transformers import AutoModel
    model = AutoModel.from_pretrained(BB_DIR)
    model.eval()
    pre = json.load(open(os.path.join(BB_DIR, "preprocessor_config.json")))
    mean, std = pre.get("image_mean"), pre.get("image_std")
    print(f"    preprocessor size={pre.get('size')} mean={mean} std={std}")
    x = torch.randint(0, 255, (2, IMG, IMG, 3), dtype=torch.uint8)
    f = x.permute(0, 3, 1, 2).float() / 255.0
    f = (f - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    with torch.no_grad():
        out = model(pixel_values=f)
    toks = out.last_hidden_state
    n_patch = (IMG // 16) ** 2
    # DINOv3 w/ registers: [CLS, reg x4, patch x196] -> 201 tokens @224
    good &= check("backbone forward",
                  toks.shape[0] == 2 and toks.shape[-1] == 768 and toks.shape[1] > n_patch,
                  f"{tuple(toks.shape)} = 1 CLS + {toks.shape[1]-1-n_patch} regs + {n_patch} patches")

    print("\nALL GREEN" if good else "\nSOME CHECKS FAILED")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
