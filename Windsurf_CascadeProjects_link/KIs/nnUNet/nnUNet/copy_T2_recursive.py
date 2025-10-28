import shutil
from pathlib import Path


def main():
    src = Path("/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/T2_Bilder")
    dst = Path("/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/300_T2_Bilder")

    if not src.is_dir():
        raise FileNotFoundError(f"Quellordner nicht gefunden: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    # Alle Dateien rekursiv finden, dann case-insensitive auf .nii / .nii.gz filtern
    nifti_files = []
    try:
        for p in src.rglob("*"):
            try:
                if p.is_file():
                    name = p.name.lower()
                    if name.endswith(".nii") or name.endswith(".nii.gz"):
                        nifti_files.append(p)
            except PermissionError:
                # Überspringe Dateien/Verzeichnisse ohne Berechtigung
                continue
    except PermissionError:
        # Oberes Verzeichnis ohne Berechtigung (macOS Privacy). Nutzerhinweis erfolgt unten.
        pass

    total = len(nifti_files)
    copied = 0
    skipped = 0

    # Preview der ersten 10 gefundenen Pfade
    preview = nifti_files[:10]
    if preview:
        print("Beispielliste gefundener NIfTI-Dateien (max. 10):")
        for p in preview:
            print(f" - {p}")

    for src_file in nifti_files:
        if not src_file.is_file():
            continue
        target = dst / src_file.name

        # Nicht überschreiben: falls bereits vorhanden -> überspringen
        if target.exists():
            skipped += 1
            continue

        shutil.copy2(src_file, target)
        copied += 1

    print(f"Gefunden:  {total} NIfTI-Dateien in '{src}'")
    print(f"Kopiert:   {copied} -> '{dst}'")
    print(f"Übersprungen (bereits vorhanden): {skipped}")
    if total == 0:
        print("Hinweis: Wenn du macOS Privacy-Permissions siehst, gib deiner IDE/Terminal 'Full Disk Access' für den Desktop oder verschiebe den Ordner an einen frei zugänglichen Ort.")


if __name__ == "__main__":
    main()
