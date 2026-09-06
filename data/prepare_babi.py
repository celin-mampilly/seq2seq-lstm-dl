import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DOWNLOAD bAbI DATASET
# ============================================================

DATASET_URL = (
    "https://s3.amazonaws.com/text-datasets/"
    "babi_tasks_1-20_v1-2.tar.gz"
)

ARCHIVE_PATH = RAW_DIR / "tasks_1-20_v1-2.tar.gz"

print("=" * 60)
print("bAbI Dataset Preparation")
print("=" * 60)


if not ARCHIVE_PATH.exists():

    print("\nDownloading bAbI dataset...")
    print("This may take a little while.\n")

    urllib.request.urlretrieve(
        DATASET_URL,
        ARCHIVE_PATH
    )

    print("Download complete!")

else:

    print("\nDataset archive already exists.")
    print("Skipping download.")


# ============================================================
# 3. EXTRACT DATASET
# ============================================================

EXTRACT_DIR = RAW_DIR / "babi_dataset"

if not EXTRACT_DIR.exists():

    print("\nExtracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)

    print("Extraction complete!")

else:

    print("\nDataset already extracted.")
    print("Skipping extraction.")


# ============================================================
# 4. FIND TASK 1 FILES
# ============================================================

train_file = (
    EXTRACT_DIR
    / "tasks_1-20_v1-2"
    / "en-10k"
    / "qa1_single-supporting-fact_train.txt"
)

test_file = (
    EXTRACT_DIR
    / "tasks_1-20_v1-2"
    / "en-10k"
    / "qa1_single-supporting-fact_test.txt"
)


if not train_file.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{train_file}"
    )

if not test_file.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{test_file}"
    )


print("\nTask 1 files found!")
print("Train:", train_file)
print("Test :", test_file)


# ============================================================
# 5. PARSE bAbI FILE
# ============================================================

def parse_babi_file(file_path):

    rows = []
    context = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            number, content = line.split(" ", 1)

            # New story
            if number == "1":
                context = []

            # Question line
            if "\t" in content:

                question, answer, supporting_id = content.split(
                    "\t"
                )

                context_text = " ".join(context)

                rows.append({
                    "context": context_text,
                    "question": question,
                    "answer": answer
                })

            else:

                # Context sentence
                context.append(content)

    return pd.DataFrame(rows)


# ============================================================
# 6. CONVERT TRAIN AND TEST
# ============================================================

print("\n" + "=" * 60)
print("Converting bAbI Dataset")
print("=" * 60)

train_df = parse_babi_file(train_file)
test_df = parse_babi_file(test_file)

print("\nOriginal dataset:")
print("Train samples:", len(train_df))
print("Test samples :", len(test_df))


# ============================================================
# 7. CREATE VALIDATION SET
# ============================================================

print("\nCreating validation set...")

train_df, validation_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 8. CLEAN TEXT
# ============================================================

def clean_dataframe(df):

    df = df.copy()

    for column in ["context", "question", "answer"]:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return df


train_df = clean_dataframe(train_df)
validation_df = clean_dataframe(validation_df)
test_df = clean_dataframe(test_df)


# ============================================================
# 9. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("\nTraining   :", len(train_df))
print("Validation :", len(validation_df))
print("Testing    :", len(test_df))

print("\nColumns:")
print(list(train_df.columns))


# ============================================================
# 10. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE")
print("=" * 60)

sample = train_df.iloc[0]

print("\nContext:")
print(sample["context"])

print("\nQuestion:")
print(sample["question"])

print("\nAnswer:")
print(sample["answer"])


# ============================================================
# 11. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nTrain:")
print(train_df.isnull().sum())

print("\nValidation:")
print(validation_df.isnull().sum())

print("\nTest:")
print(test_df.isnull().sum())


# ============================================================
# 12. SAVE CSV FILES
# ============================================================

print("\n" + "=" * 60)
print("SAVING CSV FILES")
print("=" * 60)

train_path = PROCESSED_DIR / "train.csv"
validation_path = PROCESSED_DIR / "validation.csv"
test_path = PROCESSED_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


print("\nCSV files created successfully!")

print("\nTrain:")
print(train_path)

print("\nValidation:")
print(validation_path)

print("\nTest:")
print(test_path)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE!")
print("=" * 60)

print("""
Your processed dataset is now:

data/
├── raw/
│   └── ...
│
└── processed/
    ├── train.csv
    ├── validation.csv
    └── test.csv
""")