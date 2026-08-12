r"""Run box-based ITA over a sample of 140k (or ANY image folder) and report
the skin-tone distribution. No training — just measurement.

Usage (point it at wherever 140k lives):
  python check_140k_tone.py --root "C:\path\to\140k" --sample 2000

It walks the folder, detects a face in each sampled image, measures ITA in
the face region, and prints the tone breakdown + copies sample crops per
bucket so you can eyeball whether 'dark' is actually dark.
"""
import argparse, sys, shutil
from pathlib import Path
def _bootstrap():
    """Find the sadf package whether it's under src/ or at the repo root."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    print("ERROR: could not find the 'sadf' package. Run this from your "
          "project root (the folder containing scripts/ and sadf/ or src/sadf/).")
    sys.exit(1)
_bootstrap()
import numpy as np, cv2
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x
from sadf.data.faces import FaceCropper
from sadf.data.skin_tone import estimate_ita
from sadf.data.pilot import coarsen_tone

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing 140k images")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--out", default="data/_140k_tone_check")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Folder not found: {root}"); sys.exit(1)

    print(f"Scanning {root} for images (this can take a moment for 140k)...", flush=True)
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(all_imgs)} images. Sampling {min(args.sample,len(all_imgs))}...\n")
    if not all_imgs:
        print("No images found. Check the --root path and folder structure."); sys.exit(1)

    rng = np.random.default_rng(0)
    sample = [all_imgs[i] for i in rng.permutation(len(all_imgs))[:args.sample]]
    cropper = FaceCropper("haar", min_face=40, margin=0.30)

    out = Path(args.out)
    if out.exists(): shutil.rmtree(out)
    counts = {"light":0,"medium":0,"dark":0,"unknown":0}
    no_face = 0
    saved = {k:0 for k in counts}

    print('Measuring faces...', flush=True)
    for p in tqdm(sample):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None: continue
        crop, found, box = cropper.crop_with_box(bgr)
        if crop is None:
            no_face += 1; continue
        ita, fine, sf = estimate_ita(crop, face_box=box)
        tone = coarsen_tone(fine)
        if tone not in counts: tone = "unknown"
        counts[tone] += 1
        if saved[tone] < 15:
            d = out/tone; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d/f"{saved[tone]}_ita{ita:.0f}.jpg"), crop)
            saved[tone] += 1

    total = sum(counts.values())
    print("=== 140k skin-tone distribution (box-based ITA) ===")
    for k in ["light","medium","dark","unknown"]:
        pct = counts[k]/total*100 if total else 0
        print(f"  {k:8s}: {counts[k]:5d}  ({pct:.1f}%)")
    print(f"  (no face detected: {no_face} images skipped)")
    print(f"\nSample crops per bucket in {out}\\<tone>\\ — EYEBALL the dark/ folder.")
    print("If dark/ is real dark-skinned faces -> 140k is more diverse, switch to it.")
    print("If dark/ is still light faces -> both datasets are light-skewed (finding).")


if __name__ == "__main__":
    main()