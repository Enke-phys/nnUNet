import os
import sys
import glob
import shutil
import random
import re
from collections import defaultdict

# ===== Konfiguration (einfach anpassen) =====
IMAGES_DIR = "/Pfad/zu/deinen/MRI_Bildern"     # z.B. "/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/images_all"
LABELS_DIR = "/Pfad/zu/deinen/Label_Bildern"   # z.B. "/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/labels_all"
OUT_ROOT   = "/Pfad/zu/Ausgabe_4_Ordnern"      # z.B. "/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT"
SPLIT_RATIO = 0.8   # 0.8 = 80% Train, 20% Test
SEED = 42           # für reproduzierbares Zufalls-Splitting
MOVE_FILES = False  # True = verschieben, False = kopieren
# ============================================

def stem(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]

def file_ext(path):
    # Gibt ".nii.gz" oder ".nii" zurück (oder echte Extension)
    name = os.path.basename(path).lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    return os.path.splitext(name)[1]

def image_key_and_channel(path):
    """
    Liefert:
      - key: Fall-ID ohne Kanal-Suffix (z.B. 'leg_0001' für 'leg_0001_0000.nii.gz')
      - ch:  vierstelliger Kanalindex als String ('0000', '0001', …). Falls keiner vorhanden: '0000'
    """
    s = stem(path)
    m = re.match(r"^(.*)_(\d{4})$", s)
    if m:
        return m.group(1), m.group(2)
    else:
        # kein Kanal im Namen -> behandle als Ein-Kanal '_0000'
        return s, "0000"

def list_nii(dir_path):
    return sorted([p for p in glob.glob(os.path.join(dir_path, "*"))
                   if p.lower().endswith(".nii") or p.lower().endswith(".nii.gz")])

def main():
    if not os.path.isdir(IMAGES_DIR):
        print(f"Fehler: IMAGES_DIR existiert nicht: {IMAGES_DIR}")
        sys.exit(1)
    if not os.path.isdir(LABELS_DIR):
        print(f"Fehler: LABELS_DIR existiert nicht: {LABELS_DIR}")
        sys.exit(1)

    imgs = list_nii(IMAGES_DIR)
    labs = list_nii(LABELS_DIR)

    if not imgs:
        print("Keine MRI-Dateien gefunden.")
        sys.exit(1)
    if not labs:
        print("Keine Label-Dateien gefunden.")
        sys.exit(1)

    # Bilder nach Fall gruppieren (key ohne Kanal), Kanäle behalten
    img_groups = defaultdict(list)  # key -> Liste (pfad, kanal, ext)
    for p in imgs:
        k, ch = image_key_and_channel(p)
        img_groups[k].append((p, ch, file_ext(p)))

    # Labels nach Fall (ohne Kanal)
    lab_map = {stem(p): p for p in labs}

    # Gemeinsame Fälle (nur wo Bild+Label existiert)
    common_keys = sorted(set(img_groups.keys()) & set(lab_map.keys()))
    only_imgs = sorted(set(img_groups.keys()) - set(lab_map.keys()))
    only_labs = sorted(set(lab_map.keys()) - set(img_groups.keys()))

    if only_imgs:
        print(f"Warnung: Für diese Bild-Fälle fehlt ein Label: {only_imgs}")
    if only_labs:
        print(f"Warnung: Für diese Label-Fälle fehlt ein Bild: {only_labs}")

    if not common_keys:
        print("Keine übereinstimmenden Bild/Label-Paare gefunden.")
        sys.exit(1)

    # Zufälliges Splitting (Fälle/keys)
    random.seed(SEED)
    random.shuffle(common_keys)
    split_idx = int(len(common_keys) * SPLIT_RATIO)
    train_keys = common_keys[:split_idx]
    test_keys  = common_keys[split_idx:]

    # Ausgabeordner
    imagesTr = os.path.join(OUT_ROOT, "imagesTr")
    imagesTs = os.path.join(OUT_ROOT, "imagesTs")
    labelsTr = os.path.join(OUT_ROOT, "labelsTr")
    labelsTs = os.path.join(OUT_ROOT, "labelsTs")
    for d in [imagesTr, imagesTs, labelsTr, labelsTs]:
        os.makedirs(d, exist_ok=True)

    # Kopier-/Verschiebe-Funktion
    op = shutil.move if MOVE_FILES else shutil.copy2

    def transfer(keys, img_out_dir, lab_out_dir):
        for k in keys:
            # Bilder: alle Kanäle des Falls nach nnUNet-Schema in images*/ kopieren
            for src_img, ch, ext in sorted(img_groups[k], key=lambda t: t[1]):
                dst_img = os.path.join(img_out_dir, f"{k}_{ch}{ext}")
                op(src_img, dst_img)
            # Label: ohne Kanal-Suffix nach labels*/ kopieren (Extension beibehalten)
            src_lab = lab_map[k]
            lab_ext = file_ext(src_lab)
            dst_lab = os.path.join(lab_out_dir, f"{k}{lab_ext}")
            op(src_lab, dst_lab)

    transfer(train_keys, imagesTr, labelsTr)
    transfer(test_keys,  imagesTs, labelsTs)

    print("Fertig.")
    print(f"Fälle gesamt: {len(common_keys)}  -> Train: {len(train_keys)}, Test: {len(test_keys)}")
    print(f"Ausgabeordner: {OUT_ROOT}")
    print("Befüllt: imagesTr, imagesTs, labelsTr, labelsTs")

if __name__ == "__main__":
    main()