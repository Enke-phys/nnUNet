import shutil
from pathlib import Path

def main():
    src = Path("/Users/enke/Documents/nnUNetDataset/nnUNet_raw/Dataset001_MRT/10_DatenEssen")
    dst = src.parent / "300_T1_Bilder"
    dst.mkdir(parents=True, exist_ok=True)

    suffixes = ("_OS_T1.nii.gz", "_US_T1.nii.gz")

    total = 0
    copied = 0
    skipped = 0

    for f in src.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if any(name.endswith(suf) for suf in suffixes):
            total += 1
            target = dst / name
            if target.exists():
                skipped += 1
                # Wenn Überschreiben gewünscht: shutil.copy2(f, target)
                continue
            shutil.copy2(f, target)
            copied += 1

    print(f"Gefunden: {total} passende Dateien")
    print(f"Kopiert:  {copied} -> {dst}")
    print(f"Übersprungen (bereits vorhanden): {skipped}")

if __name__ == "__main__":
    main()