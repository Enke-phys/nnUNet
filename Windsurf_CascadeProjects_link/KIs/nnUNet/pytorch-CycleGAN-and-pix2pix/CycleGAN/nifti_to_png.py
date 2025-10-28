import os
import nibabel as nib
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count, set_start_method

def get_filename_without_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def process_nifti(nifti_path: str):
    try:
        img = nib.load(nifti_path).get_fdata(dtype=np.float32)

        # Normalisierung auf 0-255 für PNG (numerisch stabil)
        vmin = img.min()
        vmax = img.max()
        scale = 255.0 / (vmax - vmin + 1e-8)
        img = ((img - vmin) * scale).astype(np.uint8)

        folder = os.path.dirname(nifti_path)
        base = os.path.basename(nifti_path)
        base_no_ext = get_filename_without_ext(base)

        for i in range(img.shape[2]):  # Z-Achse = Slices
            slice_img = Image.fromarray(img[:, :, i])
            slice_filename = f"{base_no_ext}_slice{i}.png"
            slice_path = os.path.join(folder, slice_filename)
            slice_img.save(slice_path)

        return (nifti_path, "ok")
    except Exception as e:
        return (nifti_path, f"error: {e}")


def collect_all_files(folders):
    all_paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
        all_paths.extend([os.path.join(folder, f) for f in files])
    return all_paths


if __name__ == "__main__":
    # macOS: sichere Startmethode und Threads begrenzen
    set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    folders = [
        "/Users/enke/Documents/CycleGAN/datasets/t1t2/trainA",
        "/Users/enke/Documents/CycleGAN/datasets/t1t2/trainB",
        "/Users/enke/Documents/CycleGAN/datasets/t1t2/testA",
        "/Users/enke/Documents/CycleGAN/datasets/t1t2/testB",
    ]

    all_nifti = collect_all_files(folders)
    print(f"Insgesamt {len(all_nifti)} NIfTI-Dateien gefunden.")

    workers = min(cpu_count(), 6)
    print(f"Starte mit {workers} Prozessen…")

    ok = 0
    fail = 0
    with Pool(processes=workers) as pool:
        for path, status in pool.imap_unordered(process_nifti, all_nifti, chunksize=4):
            if status == "ok":
                ok += 1
            else:
                fail += 1
                print(f"Fehler bei {path}: {status}")

    print(f"Fertig. Erfolgreich: {ok}, Fehler: {fail}")
