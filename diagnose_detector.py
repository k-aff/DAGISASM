"""Is the detector broken, or just the app? Run from project root:
   python diagnose_detector.py
Tests the trained model directly on your own labelled test-split images.
"""
import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import torch, numpy as np, pandas as pd
from PIL import Image
from torchvision import transforms as T
from sadf.config import _dotify
from sadf.models.detector import build_model

CKPT = "runs/deepdetect/baseline/best.pt"
MAN = "data/deepdetect/manifests/manifest.csv"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = _dotify(ckpt["config"])
model = build_model(cfg, n_groups=len(ckpt["group_vocab"]))
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded {CKPT} | epoch {ckpt.get('epoch')} | saved val: {ckpt.get('val')}")
print(f"threshold in cfg: {cfg.train.threshold}")

tf = T.Compose([T.Resize((int(cfg.image.size),)*2), T.ToTensor(),
                T.Normalize(cfg.image.mean, cfg.image.std)])

df = pd.read_csv(MAN)
test = df[(df["status"]=="ok") & (df["split"]=="test")]
print(f"\nTesting on {len(test)} held-out images...")

# sample 40 real + 40 fake, run raw model, look at score distribution
rng = np.random.default_rng(0)
rows = []
for lab in (0,1):
    sub = test[test["label"]==lab]
    pick = sub.iloc[rng.permutation(len(sub))[:40]]
    for _,r in pick.iterrows():
        img = Image.open(r["path"]).convert("RGB")
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            logit = float(model(x)["logits"][0])
            prob_fake = 1/(1+np.exp(-logit))
        rows.append({"true_label":lab, "logit":logit, "prob_fake":prob_fake})
res = pd.DataFrame(rows)

print("\n=== raw model output (prob_fake) by TRUE label ===")
for lab, name in [(0,"REAL"),(1,"FAKE")]:
    s = res[res.true_label==lab]["prob_fake"]
    print(f"{name:5s} (n={len(s)}): prob_fake min={s.min():.3f} "
          f"mean={s.mean():.3f} max={s.max():.3f}")
print("\nIf FAKE images have HIGH prob_fake and REAL images LOW -> model works, app bug.")
print("If BOTH near same value -> model didn't learn / label problem.")
print("If REVERSED (fakes low, reals high) -> LABELS ARE FLIPPED.")

# accuracy at 0.5
res["pred"] = (res.prob_fake>=0.5).astype(int)
acc = (res.pred==res.true_label).mean()
print(f"\nAccuracy @0.5 on this sample: {acc:.1%}")
# what if we flip?
acc_flip = (res.pred!=res.true_label).mean()
print(f"Accuracy if predictions flipped: {acc_flip:.1%}")
if acc_flip > 0.8 and acc < 0.2:
    print(">>> LABELS ARE INVERTED — the fix is a label flip.")
