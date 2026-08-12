"""Re-diagnose tone buckets with BOX-BASED ITA on your harvested crops.
Run from project root:  python diagnose_tone.py
NOTE: this re-measures the SAVED crops. The saved crops don't store the
original face box, so this re-runs face detection ON the crop to locate
the face, then measures skin in that region. Shows whether box-based ITA
separates tones better than the old whole-crop method.
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, "src")
import pandas as pd, numpy as np, cv2
from sadf.data.faces import FaceCropper
from sadf.data.pilot import coarsen_tone
from sadf.data.skin_tone import estimate_ita

MAN = Path("data/deepdetect/manifests/pilot_manifest.csv")
if not MAN.exists():
    print(f"Manifest not found at {MAN}"); sys.exit(1)
df = pd.read_csv(MAN)
print(f"Re-measuring {len(df)} crops with BOX-BASED ITA...\n")

cropper = FaceCropper("haar", min_face=40, margin=0.0)  # detect within the crop
out = Path("data/deepdetect/_tone_check2")
if out.exists(): shutil.rmtree(out)

rng = np.random.default_rng(0)
newtone = []
for _, row in df.iterrows():
    p = row.get("path")
    if not p or not Path(p).exists():
        newtone.append("unknown"); continue
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        newtone.append("unknown"); continue
    # locate face within the already-cropped image to get a box
    _, _, box = cropper.crop_with_box(bgr)
    ita, fine, sf = estimate_ita(bgr, face_box=box)
    newtone.append(coarsen_tone(fine))

df["tone_new"] = newtone
print("=== OLD vs NEW tone distribution ===")
print(pd.crosstab(df["skin_tone"], df["tone_new"], margins=True))
print("\n=== how many changed bucket? ===")
changed = (df["skin_tone"] != df["tone_new"]).sum()
print(f"{changed}/{len(df)} crops ({changed/len(df)*100:.0f}%) changed tone bucket")

# sample crops per NEW bucket for eyeballing
for tone in ["light","medium","dark","unknown"]:
    sub = df[(df["tone_new"]==tone) & df["path"].notna()]
    if len(sub)==0: continue
    d = out/tone; d.mkdir(parents=True, exist_ok=True)
    for i,(_,r) in enumerate(sub.iloc[rng.permutation(len(sub))[:12]].iterrows()):
        s = Path(r["path"])
        if s.exists(): shutil.copy(s, d/f"{i}.jpg")
print(f"\nSample crops (NEW buckets) in {out}\\<tone>\\ — eyeball these.")