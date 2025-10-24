# batch_register_t2_to_t1_syn.py
import os
import csv
import ants

# Paths
T1_DIR   = "/Users/enke/Documents/t1"
T2_IMG   = "/Users/enke/Documents/t2/images"
T2_LBL   = "/Users/enke/Documents/t2/labels"
OUT_IMG  = "/Users/enke/Documents/t2_registered_to_t1/images"
OUT_LBL  = "/Users/enke/Documents/t2_registered_to_t1/labels"
PAIRS_CSV= "/Users/enke/Documents/t2_registered_to_t1/pairs_list.csv"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# Load pair IDs (strip extensions to be robust)
ids = []
with open(PAIRS_CSV, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        bn = row.get("basename") or next(iter(row.values()))
        if not bn:
            continue
        if bn.endswith(".nii.gz"):
            bn = bn[:-7]
        elif bn.endswith(".nii"):
            bn = bn[:-4]
        ids.append(bn)

# Simple preprocessing: winsorize + normalize
def prep(img):
    w = ants.iMath(img, "TruncateIntensity", 0.01, 0.99)
    return ants.iMath(w, "Normalize")

print(f"Total pairs: {len(ids)}")
for idx, ID in enumerate(ids, 1):
    # Prefer .nii.gz, fallback to .nii if needed
    t1p = os.path.join(T1_DIR,  ID + ".nii.gz")
    t2p = os.path.join(T2_IMG,  ID + ".nii.gz")
    lbp = os.path.join(T2_LBL,  ID + ".nii.gz")
    if not os.path.exists(t1p): t1p = os.path.join(T1_DIR, ID + ".nii")
    if not os.path.exists(t2p): t2p = os.path.join(T2_IMG, ID + ".nii")
    if not os.path.exists(lbp): lbp = os.path.join(T2_LBL, ID + ".nii")

    oi = os.path.join(OUT_IMG, ID + ".nii.gz")
    ol = os.path.join(OUT_LBL, ID + ".nii.gz")
    if os.path.exists(oi) and os.path.exists(ol):
        print(f"[{idx}/{len(ids)}] SKIP {ID} (exists)")
        continue

    try:
        # Read images
        t1 = ants.image_read(t1p)
        t2 = ants.image_read(t2p)
        lb = ants.image_read(lbp)

        # Preprocess
        fixed  = prep(t1)
        moving = prep(t2)

        # Stage 1: rigid then affine
        rig = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        aff = ants.registration(
            fixed=fixed, moving=moving,
            type_of_transform="Affine",
            initial_transform=rig["fwdtransforms"][0]
        )

        # Stage 2: SyN (conservative iterations) with affine init
        syn = ants.registration(
            fixed=fixed, moving=moving,
            type_of_transform="SyN",
            initial_transform=aff["fwdtransforms"],
            regIterations=(40, 20, 0)
        )

        xf = syn["fwdtransforms"]

        # Apply transforms: linear for image, nearestNeighbor for label
        img_warp = ants.apply_transforms(
            fixed=t1, moving=t2, transformlist=xf, interpolator="linear"
        )
        lbl_warp = ants.apply_transforms(
            fixed=t1, moving=lb, transformlist=xf, interpolator="nearestNeighbor"
        )

        # Save
        ants.image_write(img_warp, oi)
        ants.image_write(lbl_warp, ol)
        print(f"[{idx}/{len(ids)}] OK {ID}")

    except Exception as e:
        print(f"[{idx}/{len(ids)}] FAIL {ID}: {e}")