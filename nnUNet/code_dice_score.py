import os
import glob
import csv
import numpy as np
import nibabel as nib

# ====== KONFIGURATION ======
gt_dir = "/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/labelsTs"
pred_dir = "/Users/enke/Documents/nnUNetDataset/predictions"
output_csv = "/Users/enke/Documents/nnUNetDataset/dice_results.csv"
# Optional: feste Label-IDs festlegen (ohne 0). Wenn None, pro Fall aus GT∪Pred ermitteln.
fixed_label_ids = None  # z.B. [1,2,3,...] oder None
# ===========================

def load_nii(path):
    img = nib.load(path)
    arr = img.get_fdata()
    if arr.ndim == 4:
        arr = arr[..., 0]  # 4D->3D, falls nötig
    return arr

def dice_score(a, b):
    inter = np.count_nonzero(a & b)
    a_sum = np.count_nonzero(a)
    b_sum = np.count_nonzero(b)
    if a_sum == 0 and b_sum == 0:
        return 1.0  # beide leer
    if a_sum + b_sum == 0:
        return 0.0
    return 2.0 * inter / (a_sum + b_sum)

def stem(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]

# Dateien indexieren
gt_files = {stem(p): p for p in glob.glob(os.path.join(gt_dir, "*.nii*"))
            if p.endswith(".nii") or p.endswith(".nii.gz")}
pred_files = {stem(p): p for p in glob.glob(os.path.join(pred_dir, "*.nii*"))
              if p.endswith(".nii") or p.endswith(".nii.gz")}
common_keys = sorted(set(gt_files.keys()) & set(pred_files.keys()))

if not common_keys:
    raise SystemExit("Keine übereinstimmenden Fälle gefunden. Prüfe Pfade/Dateinamen.")

rows = []
summary_per_label = {}  # label_id -> Liste Dice über Fälle (nur wenn Label im GT vorkam)
summary_per_case = []   # Mean Dice über GT-Labels pro Fall

for k in common_keys:
    gt = load_nii(gt_files[k]).astype(np.int64)
    pr = load_nii(pred_files[k]).astype(np.int64)
    if gt.shape != pr.shape:
        raise SystemExit(f"Shape mismatch für Fall {k}: GT {gt.shape} vs Pred {pr.shape}")

    if fixed_label_ids is None:
        labels = sorted(set(np.unique(gt).tolist() + np.unique(pr).tolist()))
        labels = [l for l in labels if l != 0]  # Hintergrund ausschließen
    else:
        labels = list(fixed_label_ids)

    per_case_dice_gtlabels = []
    for l in labels:
        gt_mask = gt == l
        pr_mask = pr == l
        d = dice_score(gt_mask, pr_mask)
        if np.any(gt_mask):
            per_case_dice_gtlabels.append(d)
            summary_per_label.setdefault(l, []).append(d)
        rows.append({"case": k, "label_id": l, "dice": d})

    mean_gtlabels = float(np.mean(per_case_dice_gtlabels)) if per_case_dice_gtlabels else float('nan')
    rows.append({"case": k, "label_id": "MEAN_over_GT_labels", "dice": mean_gtlabels})
    summary_per_case.append(mean_gtlabels)

# Globale Zusammenfassung
rows.append({"case": "GLOBAL", "label_id": "MEAN_over_cases", "dice": float(np.nanmean(summary_per_case))})
for l, vals in sorted(summary_per_label.items()):
    rows.append({"case": "GLOBAL", "label_id": f"label_{l}_MEAN", "dice": float(np.mean(vals))})

# CSV schreiben
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case", "label_id", "dice"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"CSV gespeichert: {output_csv}")
print("Fälle:", len(common_keys))
print("Fälle gematcht:", ", ".join(common_keys))
