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

        # process each action unit column
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

    # start training
    table = pd.DataFrame(records)

    table = table.groupby(
        ["subject", "label"], as_index=False
    ).mean(numeric_only=True)

    print("[TRAIN] Participants:", len(table))
    print("[TRAIN] Class distribution:")
    print(table["label"].value_counts())

    x = table.drop(columns=["subject", "label"])
    y = table["label"]

    # entire participants are kept out of training
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    # keep features sufficiently represented in training
    columns = x_train.columns[
        x_train.notna().mean() >= 0.6
    ]

    x_train = x_train[columns]
    x_test = x_test[columns]

    # initialize model
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        )
    )

    model.fit(x_train, y_train)

    # evaluate model on test set
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    # output results
    print("[TRAIN] Classification results:")
    print(classification_report(
            y_test, predictions,
            target_names=["Healthy control", "Parkinson's"],
            zero_division=0,
    ))

    print("[TRAIN] Balanced accuracy:", balanced_accuracy_score(y_test, predictions))
    print("[TRAIN] ROC AUC:", roc_auc_score(y_test, probabilities))
    print("[TRAIN] Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # dump training to file
    joblib.dump(
        {"model": model, "features": list(columns)},
        "model/hypomimia_baseline.joblib",
    )

    print("[TRAIN] Model saved.")
