# =========================
# IMPORTS
# =========================

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import time
import os
import re
import warnings
import pathlib

warnings.filterwarnings("ignore")

# =========================
# NLTK SETUP (IMPORTANT FIX)
# =========================

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import emoji


# =========================
# MLFLOW SAFE SETUP (FIXED)
# =========================

BASE_DIR = pathlib.Path(__file__).parent.resolve()
MLRUNS_DIR = BASE_DIR / "mlruns"

MLRUNS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
mlflow.set_experiment("FLIPKART_SENTIMENT_ANALYSIS")


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data.csv")

df = df[["Review text", "Ratings"]].dropna().drop_duplicates()


# =========================
# TEXT CLEANING
# =========================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}


def clean_text(text):

    text = str(text).lower()

    text = emoji.replace_emoji(text, replace="")

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = [
        stemmer.stem(w)
        for w in text.split()
        if w not in stop_words
    ]

    return " ".join(words)


df["clean_review"] = df["Review text"].apply(clean_text)


# =========================
# SENTIMENT LABEL
# =========================

df["sentiment"] = df["Ratings"].apply(lambda r: 0 if r <= 2 else 1)

X = df["clean_review"]
y = df["sentiment"]


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# =========================
# CV SETUP
# =========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================
# PIPELINE
# =========================

def build_pipeline(model):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        ("model", model)
    ])


# =========================
# OPTUNA OBJECTIVES
# =========================

def objective_lr(trial):

    pipeline = build_pipeline(

        LogisticRegression(
            C=trial.suggest_float("C", 0.01, 2.0, log=True),
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        )
    )

    pipeline.set_params(

        tfidf__max_features=trial.suggest_int(
            "max_features", 2000, 5000, step=500
        ),

        tfidf__min_df=trial.suggest_int("min_df", 3, 7),

        tfidf__max_df=trial.suggest_float("max_df", 0.7, 0.9)
    )

    return cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1
    ).mean()


def objective_nb(trial):

    pipeline = build_pipeline(

        MultinomialNB(
            alpha=trial.suggest_float("alpha", 0.01, 1.0, log=True)
        )
    )

    pipeline.set_params(

        tfidf__max_features=trial.suggest_int(
            "max_features", 3000, 10000, step=1000
        ),

        tfidf__min_df=trial.suggest_int("min_df", 3, 7),

        tfidf__max_df=trial.suggest_float("max_df", 0.7, 0.9)
    )

    return cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1
    ).mean()


def objective_svm(trial):

    pipeline = build_pipeline(

        LinearSVC(
            C=trial.suggest_float("C", 0.01, 2.0, log=True),
            class_weight="balanced",
            max_iter=5000
        )
    )

    pipeline.set_params(

        tfidf__max_features=trial.suggest_int(
            "max_features", 3000, 10000, step=1000
        ),

        tfidf__min_df=trial.suggest_int("min_df", 3, 7),

        tfidf__max_df=trial.suggest_float("max_df", 0.7, 0.9)
    )

    return cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1
    ).mean()


objectives = {
    "LogisticRegression": objective_lr,
    "NaiveBayes": objective_nb,
    "LinearSVM": objective_svm
}


# =========================
# TRAINING LOOP
# =========================

if __name__ == "__main__":

    results = {}

    for model_name, obj_fn in objectives.items():

        print(f"\n--- Optimizing {model_name} ---")

        with mlflow.start_run(run_name=model_name):

            study = optuna.create_study(direction="maximize")

            start = time.time()

            study.optimize(obj_fn, n_trials=20)

            fit_time = time.time() - start

            best = study.best_params


            # Build best model
            if model_name == "LogisticRegression":

                model = LogisticRegression(
                    C=best["C"],
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42
                )

            elif model_name == "NaiveBayes":

                model = MultinomialNB(alpha=best["alpha"])

            else:

                model = LinearSVC(
                    C=best["C"],
                    class_weight="balanced",
                    max_iter=5000
                )


            pipeline = build_pipeline(model)

            pipeline.set_params(
                tfidf__max_features=best["max_features"],
                tfidf__min_df=best["min_df"],
                tfidf__max_df=best["max_df"]
            )


            # Train final model
            pipeline.fit(X_train, y_train)


            # Metrics
            train_f1 = f1_score(
                y_train,
                pipeline.predict(X_train),
                average="macro"
            )

            test_f1 = f1_score(
                y_test,
                pipeline.predict(X_test),
                average="macro"
            )


            # Log to MLflow
            mlflow.log_params(best)

            mlflow.log_metric("cv_f1_macro", study.best_value)
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("fit_time", fit_time)


            # Save model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model"
            )


            results[model_name] = (
                study.best_value,
                train_f1,
                test_f1
            )


    # =========================
    # SUMMARY
    # =========================

    print("\n--- FINAL SUMMARY ---")

    for model, (cv, tr, te) in results.items():

        print(
            f"{model} | CV={cv:.4f} | "
            f"Train={tr:.4f} | Test={te:.4f}"
        )
