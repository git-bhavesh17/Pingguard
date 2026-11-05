import re
import json
import zipfile
import urllib.request
import inspect
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "static" / "images"

for directory in [DATASET_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. RULE-BASED WARNING SIGN ENGINE (Used by Real-Time Web App)
# =====================================================================
WARNING_SIGN_RULES = [
    {
        "id": "suspicious_link",
        "title": "Link or Shortened URL Pattern",
        "pattern": re.compile(
            r"(https?://\S+|www\.\S+|\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|rb\.gy)/\S+|\b[a-zA-Z0-9.-]+\.(?:com|net|org|info|biz|co\.uk|in|xyz|top|club|online|site)\b)",
            re.IGNORECASE,
        ),
        "explanation": "Contains a web link or domain pattern. Phishing SMS messages frequently use external or shortened URLs to hide malicious destinations.",
    },
    {
        "id": "urgent_pressure",
        "title": "Urgent or High-Pressure Language",
        "pattern": re.compile(
            r"\b(urgent|immediately|act now|final notice|last chance|expires today|within \d+ hours|immediate action|account suspended|blocked)\b",
            re.IGNORECASE,
        ),
        "explanation": "Uses time-pressure wording designed to rush the recipient into acting before verifying the sender.",
    },
    {
        "id": "prize_lottery",
        "title": "Prize, Lottery, or Unclaimed Reward Claim",
        "pattern": re.compile(
            r"\b(congratulations|you have won|winner|claim your prize|cash prize|lottery|selected to receive|free gift|guaranteed reward|awarded)\b",
            re.IGNORECASE,
        ),
        "explanation": "Promises unexpected winnings or rewards, a common lure used in lottery and advance-fee scams.",
    },
    {
        "id": "credential_otp_request",
        "title": "Sensitive Credential, OTP, or Banking Request",
        "pattern": re.compile(
            r"\b(otp|one time password|pin number|share your pin|cvv|bank account|verify your account|update kyc|pan card|password|login credentials|security code)\b",
            re.IGNORECASE,
        ),
        "explanation": "Requests or references sensitive authentication or banking details. Legitimate organizations do not ask for OTPs or PINs via SMS.",
    },
    {
        "id": "financial_payment_lure",
        "title": "Financial / Currency / Premium Callback Pattern",
        "pattern": re.compile(
            r"([£$€₹]\s?\d+|\b\d+\s?(?:gbp|usd|inr|rs)\b|\bcall\s+09\d{8,10}\b|\btext\s+[A-Z0-9]+\s+to\s+\d{4,6}\b|\bunauthorized transaction\b)",
            re.IGNORECASE,
        ),
        "explanation": "References specific monetary amounts, premium-rate numbers, or shortcode reply instructions commonly seen in financial spam.",
    },
]


def analyze_warning_signs(message_text: str) -> list:
    if not isinstance(message_text, str) or not message_text.strip():
        return []

    detected = []
    for rule in WARNING_SIGN_RULES:
        match = rule["pattern"].search(message_text)
        if match:
            detected.append(
                {
                    "id": rule["id"],
                    "title": rule["title"],
                    "matched_text": match.group(0),
                    "explanation": rule["explanation"],
                }
            )
    return detected


# =====================================================================
# 2. DATASET ACQUISITION AND PREPARATION
# =====================================================================
def load_raw_dataset() -> pd.DataFrame:
    local_tsv = DATASET_DIR / "SMSSpamCollection"
    local_csv = DATASET_DIR / "sms_spam_raw.csv"
    alt_csv = DATASET_DIR / "spam.csv"

    if local_tsv.exists():
        print(f"[INFO] Loading dataset from local file: {local_tsv}")
        return pd.read_csv(
            local_tsv, sep="\t", header=None, names=["label", "message"], quoting=3, encoding="utf-8"
        )

    if local_csv.exists():
        print(f"[INFO] Loading dataset from local CSV: {local_csv}")
        return pd.read_csv(local_csv)

    if alt_csv.exists():
        print(f"[INFO] Loading dataset from local CSV: {alt_csv}")
        return pd.read_csv(alt_csv, encoding="latin-1")

    # Check if user dropped any other CSV or TSV file into dataset/
    for fpath in DATASET_DIR.glob("*.csv"):
        print(f"[INFO] Loading custom CSV from dataset folder: {fpath}")
        return pd.read_csv(fpath, encoding="latin-1")

    try:
        from ucimlrepo import fetch_ucirepo

        print("[INFO] Fetching UCI SMS Spam Collection (id=228) via ucimlrepo...")
        sms_repo = fetch_ucirepo(id=228)
        df_raw = pd.concat([sms_repo.data.targets, sms_repo.data.features], axis=1)
        df_raw.to_csv(local_csv, index=False)
        return df_raw
    except Exception as e:
        print(f"[WARN] ucimlrepo fetch failed ({e}). Attempting direct UCI archive download...")

    uci_zip_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = DATASET_DIR / "sms_spam_collection.zip"
    urllib.request.urlretrieve(uci_zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("SMSSpamCollection", path=DATASET_DIR)
    df_raw = pd.read_csv(
        local_tsv, sep="\t", header=None, names=["label", "message"], quoting=3, encoding="utf-8"
    )
    df_raw.to_csv(local_csv, index=False)
    return df_raw


def inspect_and_prepare_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    label_col, msg_col = None, None
    for col in df.columns:
        unique_vals = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
        if unique_vals.issubset({"ham", "spam", "0", "1"}) and len(unique_vals) >= 2:
            label_col = col
        elif msg_col is None:
            msg_col = col

    if label_col is None or msg_col is None:
        label_col, msg_col = df.columns[0], df.columns[1]

    df = df[[label_col, msg_col]].rename(columns={label_col: "label", msg_col: "message"}).copy()
    raw_row_count = int(len(df))
    missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"0": "ham", "1": "spam", "ham": "ham", "spam": "spam"})
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].copy()

    duplicate_count = int(df.duplicated(subset=["label", "message"]).sum())
    df_clean = df.drop_duplicates(subset=["label", "message"], keep="first").reset_index(drop=True)
    df_clean["target"] = (df_clean["label"] == "spam").astype(int)
    df_clean["char_length"] = df_clean["message"].str.len()
    df_clean["word_count"] = df_clean["message"].str.split().str.len()

    cleaned_row_count = int(len(df_clean))
    class_counts = df_clean["label"].value_counts().to_dict()
    ham_count = int(class_counts.get("ham", 0))
    spam_count = int(class_counts.get("spam", 0))
    ham_pct = round((ham_count / cleaned_row_count) * 100, 2)
    spam_pct = round((spam_count / cleaned_row_count) * 100, 2)

    length_stats = {}
    for lbl in ["ham", "spam"]:
        sub = df_clean[df_clean["label"] == lbl]
        length_stats[lbl] = {
            "avg_char_length": round(float(sub["char_length"].mean()), 2),
            "median_char_length": round(float(sub["char_length"].median()), 2),
            "avg_word_count": round(float(sub["word_count"].mean()), 2),
            "median_word_count": round(float(sub["word_count"].median()), 2),
        }

    prep_summary = {
        "raw_rows": raw_row_count,
        "missing_values": missing_values,
        "duplicate_rows_removed": duplicate_count,
        "cleaned_rows": cleaned_row_count,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_percentage": ham_pct,
        "spam_percentage": spam_pct,
        "length_statistics": length_stats,
    }
    return df_clean, prep_summary


# =====================================================================
# 3. TEXT PREPROCESSING & EDA PLOTS
# =====================================================================
NEGATION_WORDS = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "without"}
CUSTOM_STOP_WORDS = sorted(list(ENGLISH_STOP_WORDS - NEGATION_WORDS))


def preprocess_sms_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[£$€₹]", " currencytoken ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_ngrams(messages: pd.Series, n: int = 15, ngram_range=(1, 2)):
    vec = TfidfVectorizer(
        preprocessor=preprocess_sms_text,
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=ngram_range,
        max_features=2000,
    )
    analyzer = vec.build_analyzer()
    counter = Counter()
    for msg in messages:
        counter.update(analyzer(msg))
    return [{"term": term, "count": int(cnt)} for term, cnt in counter.most_common(n)]


def generate_eda_visualizations(df_clean: pd.DataFrame, prep_summary: dict) -> dict:
    sns.set_theme(style="whitegrid")
    palette = {"ham": "#14b8a6", "spam": "#f43f5e"}

    # 1. Class Count Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Ham (Legitimate)", "Spam (Suspicious)"], [prep_summary["ham_count"], prep_summary["spam_count"]], color=["#14b8a6", "#f43f5e"], width=0.5)
    ax.set_title("SMS Dataset Class Distribution (Counts)", fontweight="bold")
    ax.set_ylabel("Number of Messages")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "01_class_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Class Percentage Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Ham (%)", "Spam (%)"], [prep_summary["ham_percentage"], prep_summary["spam_percentage"]], color=["#0ea5e9", "#e11d48"], width=0.5)
    ax.set_title("SMS Class Proportions (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "02_class_percentages.png", dpi=140)
    plt.close(fig)

    # 3. Character Length Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df_clean[df_clean["char_length"] <= 350], x="char_length", hue="label", palette=palette, bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of SMS Character Lengths by Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "03_char_length_distribution.png", dpi=140)
    plt.close(fig)

    # 4. Word Count Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df_clean[df_clean["word_count"] <= 75], x="word_count", hue="label", palette=palette, bins=35, kde=True, ax=ax)
    ax.set_title("Distribution of SMS Word Counts by Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "04_word_count_distribution.png", dpi=140)
    plt.close(fig)

    # 5. Average Message Length Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["Ham", "Spam"], [prep_summary["length_statistics"]["ham"]["avg_char_length"], prep_summary["length_statistics"]["spam"]["avg_char_length"]], color=["#14b8a6", "#f43f5e"], width=0.5)
    axes[0].set_title("Average Character Length", fontweight="bold")
    axes[1].bar(["Ham", "Spam"], [prep_summary["length_statistics"]["ham"]["avg_word_count"], prep_summary["length_statistics"]["spam"]["avg_word_count"]], color=["#14b8a6", "#f43f5e"], width=0.5)
    axes[1].set_title("Average Word Count", fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "05_avg_length_comparison.png", dpi=140)
    plt.close(fig)

    top_spam_terms = get_top_ngrams(df_clean[df_clean["label"] == "spam"]["message"], n=15)
    top_ham_terms = get_top_ngrams(df_clean[df_clean["label"] == "ham"]["message"], n=15)

    bins_char = list(range(0, 320, 20))
    ham_char_hist, _ = np.histogram(df_clean[df_clean["label"] == "ham"]["char_length"].clip(upper=300), bins=bins_char)
    spam_char_hist, _ = np.histogram(df_clean[df_clean["label"] == "spam"]["char_length"].clip(upper=300), bins=bins_char)
    bin_labels = [f"{bins_char[i]}-{bins_char[i+1]}" for i in range(len(bins_char) - 1)]

    eda_summary = {
        **prep_summary,
        "top_spam_terms": top_spam_terms,
        "top_ham_terms": top_ham_terms,
        "char_length_histogram": {
            "labels": bin_labels,
            "ham": [int(x) for x in ham_char_hist],
            "spam": [int(x) for x in spam_char_hist],
        },
    }

    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    return eda_summary


# =====================================================================
# 4. MODEL TRAINING, VALIDATION SELECTION & TEST EVALUATION
# =====================================================================
def build_models():
    ada_sig = inspect.signature(AdaBoostClassifier.__init__).parameters
    ada_base_tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    ada_kwargs = {"n_estimators": 100, "learning_rate": 0.8, "random_state": RANDOM_STATE}
    if "estimator" in ada_sig:
        ada_kwargs["estimator"] = ada_base_tree
    else:
        ada_kwargs["base_estimator"] = ada_base_tree
    if "algorithm" in ada_sig:
        ada_kwargs["algorithm"] = "SAMME"

    return {
        "decision_tree": {
            "name": "Decision Tree",
            "type": "Single Learner (Baseline)",
            "description": "Single hierarchical tree using Gini impurity splits. Fast and interpretable, but higher variance than ensembles.",
            "estimator": DecisionTreeClassifier(max_depth=25, min_samples_split=5, class_weight="balanced", random_state=RANDOM_STATE),
        },
        "random_forest": {
            "name": "Random Forest",
            "type": "Bagging Ensemble",
            "description": "Bootstrap aggregation of 150 decorrelated trees with random feature subsets to reduce variance.",
            "estimator": RandomForestClassifier(n_estimators=150, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        },
        "adaboost": {
            "name": "AdaBoost",
            "type": "Adaptive Boosting Ensemble",
            "description": "Sequential shallow trees that increase weights on previously misclassified messages.",
            "estimator": AdaBoostClassifier(**ada_kwargs),
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "type": "Gradient Boosting Ensemble",
            "description": "Sequential additive model fitting each new weak tree to the negative gradient of the loss function.",
            "estimator": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=RANDOM_STATE),
        },
        "xgboost": {
            "name": "XGBoost",
            "type": "Regularized Gradient Boosting",
            "description": "Optimized second-order gradient boosting with L1/L2 regularization and column subsampling.",
            "estimator": XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        },
    }


def evaluate_single_split(estimator, X_features, y_true):
    y_pred = estimator.predict(X_features)
    y_scores = estimator.predict_proba(X_features)[:, 1] if hasattr(estimator, "predict_proba") else y_pred.astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    roc_idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr)), dtype=int)
    pr_idx = np.linspace(0, len(prec_curve) - 1, min(40, len(prec_curve)), dtype=int)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_scores)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0),
        "roc_curve": {"fpr": [round(float(fpr[i]), 4) for i in roc_idx], "tpr": [round(float(tpr[i]), 4) for i in roc_idx]},
        "pr_curve": {"recall": [round(float(rec_curve[i]), 4) for i in pr_idx], "precision": [round(float(prec_curve[i]), 4) for i in pr_idx]},
    }


def extract_feature_importances(estimator, feature_names, top_k=15):
    if not hasattr(estimator, "feature_importances_"):
        return []
    importances = estimator.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_k]
    return [
        {"feature": str(feature_names[idx]), "importance": round(float(importances[idx]), 5)}
        for idx in top_indices
        if importances[idx] > 0
    ]


def run_full_pipeline():
    print("[1/4] Loading and preparing dataset...")
    df_raw = load_raw_dataset()
    df_clean, prep_summary = inspect_and_prepare_data(df_raw)
    print(f"      Cleaned rows: {prep_summary['cleaned_rows']} (Removed {prep_summary['duplicate_rows_removed']} duplicates)")

    print("[2/4] Generating EDA visualizations and summaries...")
    eda_summary = generate_eda_visualizations(df_clean, prep_summary)

    print("[3/4] Performing 60/20/20 stratified split & fitting TF-IDF on training data...")
    X = df_clean["message"].values
    y = df_clean["target"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_sms_text,
        stop_words=CUSTOM_STOP_WORDS,
        max_features=1500,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    train_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    print("[4/4] Training and evaluating all 5 classifiers...")
    models_spec = build_models()
    model_results = {}
    comparison_rows = []

    for model_id, info in models_spec.items():
        print(f"      -> Training {info['name']}...")
        estimator = info["estimator"]
        if model_id in {"adaboost", "gradient_boosting", "xgboost"}:
            estimator.fit(X_train_tfidf, y_train, sample_weight=train_sample_weights)
        else:
            estimator.fit(X_train_tfidf, y_train)

        joblib.dump(estimator, MODELS_DIR / f"{model_id}.joblib")

        val_metrics = evaluate_single_split(estimator, X_val_tfidf, y_val)
        test_metrics = evaluate_single_split(estimator, X_test_tfidf, y_test)

        model_results[model_id] = {
            "id": model_id,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "validation": val_metrics,
            "test": test_metrics,
            "top_features": extract_feature_importances(estimator, feature_names, top_k=15),
        }

        comparison_rows.append(
            {
                "model_id": model_id,
                "model_name": info["name"],
                "model_type": info["type"],
                "val_f1": val_metrics["f1_score"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1_score"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
            }
        )

    default_model_id = max(
        model_results.keys(),
        key=lambda m: (model_results[m]["validation"]["f1_score"], model_results[m]["validation"]["pr_auc"]),
    )
    default_model_name = model_results[default_model_id]["name"]
    print(f"[DONE] Selected Default Model via Validation Spam F1: {default_model_name}")

    pd.DataFrame(comparison_rows).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    metrics_payload = {
        "default_model_id": default_model_id,
        "default_model_name": default_model_name,
        "selection_criterion": "Highest Validation Set Spam-Class F1-Score",
        "models": model_results,
        "comparison_table": comparison_rows,
    }

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return eda_summary, metrics_payload


if __name__ == "__main__":
    run_full_pipeline()
