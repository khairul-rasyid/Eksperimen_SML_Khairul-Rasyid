
# IMPORT LIBRARY

import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split

# 1. KONFIGURASI

SEED = 42
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR.parent / "dataset_5_class_animal_raw"

OUTPUT_DIR = BASE_DIR / "dataset_5_class_animal_preprocessing"

REPORT_DIR = BASE_DIR / "reports"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)

# 2. PENGUMPULAN DATASET

def collect_images():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {RAW_DIR}"
        )

    data = []

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                data.append({
                    "filepath": str(image_path.resolve()),
                    "label": label
                })

    df = pd.DataFrame(data, columns=["filepath", "label"])

    if df.empty:
        raise ValueError(
            "Tidak ada gambar yang ditemukan."
            "Periksa struktur folder dan ekstensi gambar."
        )

    print("\n========== INFORMASI DATASET ==========")
    print("Lokasi dataset:", RAW_DIR)
    print("Jumlah gambar:", len(df))
    print("Jumlah kelas:", df["label"].nunique())
    print("Nama kelas:", sorted(df["label"].unique()))

    return df


# 3. EXPLORATORY DATA ANALYSIS (EDA)

def perform_eda(df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== DISTRIBUSI KELAS ==========")
    print(df["label"].value_counts().sort_index())

    # Menyimpan distribusi kelas.
    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="jumlah_gambar")
    )

    class_counts.to_csv(
        REPORT_DIR / "distribusi_kelas.csv",
        index=False
    )

    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df,
        x="label",
        order=class_counts["label"].tolist()
    )
    plt.title("Distribusi Dataset Berdasarkan Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "distribusi_kelas.png",
        dpi=150
    )
    plt.close()

    # Memeriksa ukuran asli gambar.
    image_sizes = []

    for path in df["filepath"]:
        try:
            with Image.open(path) as image:
                image_sizes.append(image.size)
        except Exception:
            image_sizes.append(None)

    size_counts = (
        pd.Series(image_sizes, name="ukuran")
        .value_counts(dropna=False)
        .rename_axis("ukuran")
        .reset_index(name="jumlah_gambar")
    )

    size_counts.to_csv(
        REPORT_DIR / "ukuran_gambar.csv",
        index=False
    )

    print("\n========== UKURAN GAMBAR ==========")
    print(size_counts.head(20))

    # Menyimpan contoh gambar untuk setiap kelas.
    classes = sorted(df["label"].unique())
    sample_classes = classes[:5]

    fig, axes = plt.subplots(
        1,
        len(sample_classes),
        figsize=(4 * len(sample_classes), 5)
    )

    if len(sample_classes) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, sample_classes):
        sample = (
            df[df["label"] == class_name]
            .sample(n=1, random_state=SEED)
            .iloc[0]
        )

        try:
            with Image.open(sample["filepath"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(class_name)
            ax.axis("off")
        except Exception:
            ax.set_title(f"{class_name}\nGagal dibaca")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "contoh_gambar.png",
        dpi=150
    )
    plt.close()

    print("\nHasil EDA disimpan di:", REPORT_DIR)


# 4. VALIDASI DAN PEMBERSIHAN DATA

def check_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def validate_images(df):
    df = df.copy()

    df["is_valid"] = df["filepath"].apply(check_image)

    valid_count = int(df["is_valid"].sum())
    invalid_count = int((~df["is_valid"]).sum())

    print("\n========== VALIDASI GAMBAR ==========")
    print("Gambar valid:", valid_count)
    print("Gambar tidak valid:", invalid_count)

    # Simpan daftar gambar yang tidak valid sebagai laporan.
    invalid_df = df[~df["is_valid"]].copy()

    invalid_df.to_csv(
        REPORT_DIR / "gambar_tidak_valid.csv",
        index=False
    )

    # Hanya gambar valid yang diteruskan.
    df_clean = (
        df[df["is_valid"]]
        .drop(columns=["is_valid"])
        .reset_index(drop=True)
    )

    if df_clean.empty:
        raise ValueError(
            "Tidak ada gambar valid yang tersisa setelah pemeriksaan."
        )

    # Memastikan setiap kelas masih memiliki cukup gambar.
    class_counts = df_clean["label"].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Dataset harus memiliki minimal dua kelas."
        )

    if (class_counts < 2).any():
        raise ValueError(
            "Ada kelas dengan kurang dari dua gambar valid. "
            "Periksa dataset sebelum melakukan pembagian."
        )

    print("Jumlah gambar setelah pembersihan:", len(df_clean))
    print("\nDistribusi setelah pembersihan:")
    print(df_clean["label"].value_counts().sort_index())

    return df_clean


# 5. PREPROCESSING

def split_dataset(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }

    print("\n========== HASIL PEMBAGIAN DATA ==========")

    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}: {len(split_df)} gambar")
        print(split_df["label"].value_counts().sort_index())

    return splits


# 6. PENYIMPANAN DATASET HASIL PREPROCESSING

def preprocess_and_save(splits):
    # Hapus hasil lama agar file yang sudah tidak digunakan
    # tidak tertinggal dalam output terbaru.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    for split_name, split_df in splits.items():
        print(f"\nMemproses split: {split_name}")

        for _, row in split_df.iterrows():
            source = Path(row["filepath"])
            label = row["label"]

            destination_dir = OUTPUT_DIR / split_name / label
            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = destination_dir / source.name

            # Jika ada nama file yang sama dalam kelas dan split,
            # tambahkan akhiran agar gambar tidak saling menimpa.
            if destination.exists():
                destination = (
                    destination_dir /
                    f"{source.stem}_{source.parent.name}{source.suffix}"
                )

            with Image.open(source) as image:
                image = image.convert("RGB")
                image = image.resize(IMAGE_SIZE)
                image.save(destination)

            manifest.append({
                "source_filepath": str(source),
                "output_filepath": str(destination.resolve()),
                "label": label,
                "split": split_name,
                "width": IMAGE_SIZE[0],
                "height": IMAGE_SIZE[1],
                "mode": "RGB"
            })

    # Simpan catatan file yang dihasilkan.
    manifest_df = pd.DataFrame(manifest)

    print("\n========== HASIL PREPROCESSING ==========")
    print("Total gambar tersimpan:", len(manifest_df))
    print("Lokasi output:", OUTPUT_DIR)
    print("Manifest:", OUTPUT_DIR / "manifest.csv")

    return manifest_df


# 7. VALIDASI HASIL PREPROCESSING

def validate_output(manifest_df):
    print("\n========== VALIDASI OUTPUT ==========")

    if manifest_df.empty:
        raise ValueError("Tidak ada gambar hasil preprocessing.")

    errors = []

    for output_path in manifest_df["output_filepath"]:
        try:
            with Image.open(output_path) as image:
                if image.size != IMAGE_SIZE:
                    errors.append(
                        f"Ukuran tidak sesuai: {output_path}"
                    )

                if image.mode != "RGB":
                    errors.append(
                        f"Mode bukan RGB: {output_path}"
                    )

                image.verify()

        except Exception as error:
            errors.append(f"Gambar gagal divalidasi: {output_path} ({error})")

    if errors:
        print("Jumlah kesalahan:", len(errors))

        for error in errors[:10]:
            print("-", error)

        raise RuntimeError(
            "Validasi output gagal. Periksa laporan di atas."
        )

    print("Semua gambar berhasil divalidasi.")
    print("Ukuran gambar:", IMAGE_SIZE)
    print("Format warna: RGB")


# 8. FUNGSI UTAMA

def main():
    set_seed()

    print("=======AUTOMATED DATASET PREPROCESSING=======")

    df = collect_images()

    perform_eda(df)

    df_clean = validate_images(df)

    splits = split_dataset(df_clean)

    manifest_df = preprocess_and_save(splits)

    validate_output(manifest_df)

    print("========PREPROCESSING SELESAI=======")


if __name__ == "__main__":
    main()