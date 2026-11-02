# responsible for training our model
# currently we are using the dataset from Harvard, which does not
# give us the actual pictures of the face, but it does provide the measurements
# which we can use to train our model.

import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA = Path("datasets/Selfi-image-Action Units.zip")
records = []

# read facial action unit CSVs directly from the ZIP
with zipfile.ZipFile(DATA, "r") as archive:
    # for every file in the ZIP, if it ends in .csv, it is usable to
    # train, and we should collect it.
    filenames = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    print("[TRAIN] CSV files found:", len(filenames))
    print("[TRAIN] Examples:", filenames[:5])

    for filename in filenames:
        stem = Path(filename).stem

        match = re.search(
            r"_(\d+)_(HC|PD)_au$", stem, re.IGNORECASE
        )

        if not match:
            continue

        subject_id = match.group(1)
        cohort = match.group(2).upper()
        expression = stem[:match.start()].lower()

        with archive.open(filename, "r") as file:
            df = pd.read_csv(file)

        df.columns = df.columns.str.strip()

        # OpenFace style action unit intensity columns
        au_columns = [
            col for col in df.columns
            if re.fullmatch(r"AU\d{2}_r", col)
        ]

        if not au_columns:
            continue

        # exclude failed face tracking frames when flagged
        if "success" in df.columns:
            df = df[
                pd.to_numeric(
                    df["success"], errors="coerce"
                ) == 1
            ]

        if df.empty:
            continue

        row = {
            "subject": f"{cohort}:{subject_id}",
            "label": int(cohort == "PD"),
        }

        for column in au_columns:
            values = pd.to_numeric(
                df[column], errors="coerce"
            ).dropna()

            if values.empty:
                continue

            prefix = f"{expression}_{column}"
            row[f"{prefix}_mean"] = values.mean()
            row[f"{prefix}_std"] = values.std()
            row[f"{prefix}_p90"] = values.quantile(0.9)

        records.append(row)

    print("[TRAIN] Usable recordings:", len(records))

    if not records:
        raise RuntimeError("No usable recordings found.")