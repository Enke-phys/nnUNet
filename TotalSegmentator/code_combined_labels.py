import os, glob
import numpy as np
import nibabel as nib

in_dir = "/Users/enke/Documents/Test MRT/CascadeProjects/windsurf-project/nnUNet/output_ordner"
out_path = "/Users/enke/Documents/Test MRT/CascadeProjects/windsurf-project/nnUNet/combined_multilabel.nii.gz"
map_txt = "/Users/enke/Documents/Test MRT/CascadeProjects/windsurf-project/nnUNet/combined_multilabel_labels.txt"

# Alle .nii / .nii.gz Dateien alphabetisch sammeln
paths = sorted([p for p in glob.glob(os.path.join(in_dir, "*.nii*"))
                if p.endswith(".nii") or p.endswith(".nii.gz")])
if not paths:
    raise SystemExit("No NIfTI files found in input directory")

# Referenz für Shape/Affine/Header
ref = nib.load(paths[0])
ref_shape = ref.shape[:3]
affine = ref.affine
header = ref.header.copy()

# Multilabel-Array (0 = Hintergrund)
labelmap = np.zeros(ref_shape, dtype=np.uint16)
label_list = []

for idx, p in enumerate(paths, start=1):
    img = nib.load(p)
    arr = img.get_fdata()
    if arr.ndim == 4:
        arr = arr[..., 0]  # Sicherheitshalber 4D->3D
    if arr.shape != ref_shape:
        raise SystemExit(f"Shape mismatch: {os.path.basename(p)} has {arr.shape}, expected {ref_shape}")
    mask = arr > 0  # binarisieren
    # "Last-wins"-Policy: spätere Dateien überschreiben frühere Labels
    labelmap[mask] = idx
    label_list.append((idx, os.path.basename(p)))

# Schreiben der Multilabel-NIfTI
img_out = nib.Nifti1Image(labelmap, affine, header)
nib.save(img_out, out_path)

# Mapping Label-ID -> Quelldateiname
with open(map_txt, "w") as f:
    for idx, name in label_list:
        f.write(f"{idx}\t{name}\n")

print(f"Saved multilabel NIfTI to: {out_path}")
print(f"Label mapping saved to: {map_txt}")
print(f"Labels: 1..{len(paths)} (background=0)")