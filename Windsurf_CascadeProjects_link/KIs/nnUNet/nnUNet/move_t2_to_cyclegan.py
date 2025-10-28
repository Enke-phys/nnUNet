import shutil
from pathlib import Path

SRC = Path("/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/300_T2_Bilder")
DEST_TRAIN = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/trainA")
DEST_TEST = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/testA")

COUNT_TRAIN = 307
COUNT_TEST = 78


def move_files():
    if not SRC.is_dir():
        raise FileNotFoundError(f"Quelle nicht gefunden: {SRC}")
    if not DEST_TRAIN.is_dir():
        raise FileNotFoundError(f"Ziel (trainA) nicht gefunden: {DEST_TRAIN}")
    if not DEST_TEST.is_dir():
        raise FileNotFoundError(f"Ziel (testA) nicht gefunden: {DEST_TEST}")

    files = sorted([p for p in SRC.iterdir() if p.is_file() and (p.name.lower().endswith('.nii') or p.name.lower().endswith('.nii.gz'))])

    moved_train = 0
    moved_test = 0
    skipped_existing = 0

    # Zuerst trainA befüllen
    for p in files:
        if moved_train >= COUNT_TRAIN:
            break
        target = DEST_TRAIN / p.name
        if target.exists():
            skipped_existing += 1
            continue
        shutil.move(str(p), str(target))
        moved_train += 1

    # Aktualisiere verbleibende Liste für testA
    remaining = sorted([p for p in SRC.iterdir() if p.is_file() and (p.name.lower().endswith('.nii') or p.name.lower().endswith('.nii.gz'))])
    for p in remaining:
        if moved_test >= COUNT_TEST:
            break
        target = DEST_TEST / p.name
        if target.exists():
            skipped_existing += 1
            continue
        shutil.move(str(p), str(target))
        moved_test += 1

    print(f"Quelle: {SRC}")
    print(f"trainA: bewegt {moved_train}/{COUNT_TRAIN} -> {DEST_TRAIN}")
    print(f"testA:  bewegt {moved_test}/{COUNT_TEST} -> {DEST_TEST}")
    print(f"Übersprungen wegen bereits vorhanden: {skipped_existing}")


if __name__ == "__main__":
    move_files()
