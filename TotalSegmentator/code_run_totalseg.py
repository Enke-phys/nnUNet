import os
import subprocess

ROOT = "/Users/enke/Documents/nnUNetDataset/predictions/MRT_legs_51imageswolabels"

def first_nifti(case_dir):
    # suche genau eine NIfTI (.nii.gz bevorzugt)
    files = [n for n in os.listdir(case_dir) if n.lower().endswith((".nii.gz", ".nii"))]
    files.sort()
    return os.path.join(case_dir, files[0]) if files else None

def main():
    # nur direkte Unterordner (die 51 Fälle)
    for name in sorted(os.listdir(ROOT)):
        case_dir = os.path.join(ROOT, name)
        if not os.path.isdir(case_dir):
            continue
        f = first_nifti(case_dir)
        if not f:
            print(f"Überspringe (keine NIfTI gefunden): {case_dir}")
            continue

        out = os.path.join(case_dir, "segmentations")
        os.makedirs(out, exist_ok=True)

        print(f"Verarbeite: {f}")
        cmd = [
            "TotalSegmentator",
            "-i", f,
            "-o", out,
            "--task", "total_mr",
            "--ml",
            "--fast",
            "--device", "cpu",
        ]
        subprocess.run(cmd, check=True)

    print("Fertig. Ergebnisse je Fall unter: <Fallordner>/segmentations/")

if __name__ == "__main__":
    main()