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
from scipy.sparse import hstack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
# 1. RULE-BASED WARNING SIGN ENGINE (Separate from ML Classifier)
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
            detected.append({
                "id": rule["id"],
                "title": rule["title"],
                "matched_text": match.group(0),
                "explanation": rule["explanation"],
            })
    return detected


# =====================================================================
# 2. TEXT PREPROCESSING & HYBRID TF-IDF (Word + Character N-Grams)
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


def transform_hybrid_tfidf(messages, feature_bundle: dict):
    """Combines Word TF-IDF + Character N-Gram TF-IDF to catch both vocabulary and obfuscated typos."""
    X_word = feature_bundle["word_tfidf"].transform(messages)
    X_char = feature_bundle["char_tfidf"].transform(messages)
    return hstack([X_word, X_char]).tocsr()


# =====================================================================
# 3. DATASET ACQUISITION AND PREPARATION
# =====================================================================
def load_raw_dataset() -> pd.DataFrame:
    local_tsv = DATASET_DIR / "SMSSpamCollection"
    local_csv = DATASET_DIR / "sms_spam_raw.csv"
    alt_csv = DATASET_DIR / "spam.csv"

    if local_tsv.exists():
        print(f"[INFO] Loading dataset from local file: {local_tsv}")
        return pd.read_csv(local_tsv, sep="\t", header=None, names=["label", "message"], quoting=3, encoding="utf-8")
    if local_csv.exists():
        print(f"[INFO] Loading dataset from local CSV: {local_csv}")
        return pd.read_csv(local_csv)
    if alt_csv.exists():
        print(f"[INFO] Loading dataset from local CSV: {alt_csv}")
        return pd.read_csv(alt_csv, encoding="latin-1")
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
    df_raw = pd.read_csv(local_tsv, sep="\t", header=None, names=["label", "message"], quoting=3, encoding="utf-8")
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

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"0": "ham", "1": "spam", "ham": "ham", "spam": "spam"})
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

    length_stats = {}
    for lbl in ["ham", "spam"]:
        sub = df_clean[df_clean["label"] == lbl]
        length_stats[lbl] = {
            "avg_char_length": round(float(sub["char_length"].mean()), 2),
            "avg_word_count": round(float(sub["word_count"].mean()), 2),
        }

    prep_summary = {
        "raw_rows": raw_row_count,
        "duplicate_rows_removed": duplicate_count,
        "cleaned_rows": cleaned_row_count,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_percentage": round((ham_count / cleaned_row_count) * 100, 2),
        "spam_percentage": round((spam_count / cleaned_row_count) * 100, 2),
        "length_statistics": length_stats,
    }
    return df_clean, prep_summary


def get_top_ngrams(messages: pd.Series, n: int = 15, ngram_range=(1, 2)):
    vec = TfidfVectorizer(preprocessor=preprocess_sms_text, stop_words=CUSTOM_STOP_WORDS, ngram_range=ngram_range, max_features=2000)
    analyzer = vec.build_analyzer()
    counter = Counter()
    for msg in messages:
        counter.update(analyzer(msg))
    return [{"term": term, "count": int(cnt)} for term, cnt in counter.most_common(n)]


def generate_eda_visualizations(df_clean: pd.DataFrame, prep_summary: dict) -> dict:
    sns.set_theme(style="whitegrid")
    palette = {"ham": "#14b8a6", "spam": "#f43f5e"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df_clean[df_clean["char_length"] <= 350], x="char_length", hue="label", palette=palette, bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of SMS Character Lengths by Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "03_char_length_distribution.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df_clean[df_clean["word_count"] <= 75], x="word_count", hue="label", palette=palette, bins=35, kde=True, ax=ax)
    ax.set_title("Distribution of SMS Word Counts by Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "04_word_count_distribution.png", dpi=140)
    plt.close(fig)

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
# 4. UNIT III: UNSUPERVISED LEARNING (PCA + K-MEANS ELBOW METHOD)
# =====================================================================
def run_unsupervised_analysis(df_clean: pd.DataFrame, word_tfidf: TfidfVectorizer) -> dict:
    """
    Syllabus Unit III Implementation:
    1. Principal Component Analysis (PCA): Reduces TF-IDF to 2D for visual class separation.
    2. K-Means Clustering + Elbow Method: Discovers hidden spam categories automatically.
    """
    # 1. PCA 2D projection on a stratified sample (500 messages for smooth interactive scatter)
    sample_ham = df_clean[df_clean["target"] == 0].sample(n=min(300, (df_clean["target"] == 0).sum()), random_state=RANDOM_STATE)
    sample_spam = df_clean[df_clean["target"] == 1].sample(n=min(200, (df_clean["target"] == 1).sum()), random_state=RANDOM_STATE)
    sample_df = pd.concat([sample_ham, sample_spam], ignore_index=True)

    X_sample_dense = word_tfidf.transform(sample_df["message"]).toarray()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca.fit_transform(X_sample_dense)

    pca_ham = []
    pca_spam = []
    for idx, row in sample_df.iterrows():
        pt = {"x": round(float(coords_2d[idx, 0]), 4), "y": round(float(coords_2d[idx, 1]), 4)}
        if row["target"] == 1:
            pca_spam.append(pt)
        else:
            pca_ham.append(pt)

    # 2. K-Means + Elbow Method on Spam Messages
    spam_msgs = df_clean[df_clean["target"] == 1]["message"].values
    X_spam_tfidf = word_tfidf.transform(spam_msgs)
    feature_names = word_tfidf.get_feature_names_out()

    elbow_k = [2, 3, 4, 5, 6]
    elbow_inertia = []
    for k in elbow_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X_spam_tfidf)
        elbow_inertia.append(round(float(km.inertia_), 2))

    # Fit final K=3 Spam Sub-Clusters
    km_final = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = km_final.fit_predict(X_spam_tfidf)

    spam_clusters = []
    for c_id in range(3):
        centroid = km_final.cluster_centers_[c_id]
        top_idx = np.argsort(centroid)[::-1][:7]
        top_words = [str(feature_names[i]) for i in top_idx]
        count = int((cluster_labels == c_id).sum())
        spam_clusters.append({
            "cluster_id": c_id + 1,
            "message_count": count,
            "top_keywords": top_words,
        })

    return {
        "pca_explained_variance": [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_],
        "pca_points": {"ham": pca_ham, "spam": pca_spam},
        "elbow_method": {"k_values": elbow_k, "inertia": elbow_inertia},
        "spam_clusters": spam_clusters,
    }


# =====================================================================
# 5. UNIT V: Q-LEARNING ADAPTIVE SCREENING AGENT (MDP & BELLMAN UPDATE)
# =====================================================================
RL_STATES = ["low_risk", "borderline_risk", "high_risk"]
RL_ACTIONS = ["allow_ham", "quarantine_review", "block_spam"]


def get_rl_state(spam_score: float) -> str:
    if spam_score < 0.35:
        return "low_risk"
    elif spam_score < 0.70:
        return "borderline_risk"
    return "high_risk"


def train_initial_q_table(y_val, val_scores) -> dict:
    """
    Syllabus Unit V Implementation:
    Pre-trains a Q-Learning Agent (Agent, Environment, State, Action, Reward, Policy)
    on validation predictions using the Bellman Equation:
    Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
    """
    alpha = 0.15   # Learning rate
    gamma = 0.80   # Discount factor
    q_table = {s: {a: 0.0 for a in RL_ACTIONS} for s in RL_STATES}

    # Reward matrix based on true label (0=Ham, 1=Spam) vs Agent Action
    # Penalizes false-positive blocking of legitimate Ham heavily (-15) and missing Spam (-12)
    reward_map = {
        (0, "allow_ham"): 10.0,
        (0, "quarantine_review"): -2.0,
        (0, "block_spam"): -15.0,
        (1, "allow_ham"): -12.0,
        (1, "quarantine_review"): 4.0,
        (1, "block_spam"): 10.0,
    }

    for _ in range(3):
        for true_lbl, score in zip(y_val, val_scores):
            state = get_rl_state(float(score))
            for action in RL_ACTIONS:
                r = reward_map[(int(true_lbl), action)]
                max_next_q = max(q_table[state].values())
                q_table[state][action] = round(
                    q_table[state][action] + alpha * (r + gamma * max_next_q - q_table[state][action]), 3
                )

    return {
        "alpha": alpha,
        "gamma": gamma,
        "feedback_updates": 0,
        "states": RL_STATES,
        "actions": RL_ACTIONS,
        "q_values": q_table,
    }


# =====================================================================
# 6. UNIT II & IV: SUPERVISED + ENSEMBLE MODELS & THRESHOLD TUNING
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
            "name": "Decision Tree (Entropy)",
            "type": "Supervised Baseline",
            "description": "Single Decision Tree using Entropy / Information Gain splits on hybrid word + char TF-IDF features.",
            "estimator": DecisionTreeClassifier(criterion="entropy", max_depth=25, min_samples_split=5, class_weight="balanced", random_state=RANDOM_STATE),
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "type": "Supervised Linear",
            "description": "L2-regularized Logistic Regression with balanced class weights; fast and highly effective on sparse TF-IDF text.",
            "estimator": LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        },
        "svm": {
            "name": "Support Vector Machine (SVM)",
            "type": "Supervised Margin",
            "description": "Linear Support Vector Machine (maximum-margin hyperplane) wrapped in probability calibration for smooth spam scores.",
            "estimator": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE), cv=3),
        },
        "random_forest": {
            "name": "Random Forest",
            "type": "Bagging Ensemble",
            "description": "Bootstrap aggregation of 200 decorrelated trees with feature randomness to reduce variance.",
            "estimator": RandomForestClassifier(n_estimators=200, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        },
        "adaboost": {
            "name": "AdaBoost",
            "type": "Adaptive Boosting",
            "description": "Sequential shallow trees that increase weights on previously misclassified messages.",
            "estimator": AdaBoostClassifier(**ada_kwargs),
        },
        "gradient_boosting": {
            "name": "Gradient Boosting (GBM)",
            "type": "Gradient Boosting",
            "description": "Sequential additive model fitting each new weak tree to the negative gradient of log-loss.",
            "estimator": GradientBoostingClassifier(n_estimators=140, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=RANDOM_STATE),
        },
        "xgboost": {
            "name": "XGBoost",
            "type": "Regularized Boosting",
            "description": "Optimized second-order gradient boosting with L1/L2 regularization and column subsampling.",
            "estimator": XGBClassifier(n_estimators=180, learning_rate=0.08, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        },
    }


def find_optimal_threshold(y_val, val_scores) -> float:
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(0.25, 0.76, 0.01):
        preds = (val_scores >= t).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 2)


def evaluate_from_scores(y_true, y_scores, threshold: float = 0.50):
    y_pred = (y_scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    roc_idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr)), dtype=int)
    pr_idx = np.linspace(0, len(prec_curve) - 1, min(40, len(prec_curve)), dtype=int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_scores)), 4),
        "confusion_matrix": cm.tolist(),
        "false_positive_rate": round(float(cm[0, 1] / max(1, cm[0, 0] + cm[0, 1])), 4),
        "false_negative_rate": round(float(cm[1, 0] / max(1, cm[1, 0] + cm[1, 1])), 4),
        "roc_curve": {"fpr": [round(float(fpr[i]), 4) for i in roc_idx], "tpr": [round(float(tpr[i]), 4) for i in roc_idx]},
        "pr_curve": {"recall": [round(float(rec_curve[i]), 4) for i in pr_idx], "precision": [round(float(prec_curve[i]), 4) for i in pr_idx]},
    }


def extract_feature_importances(estimator, feature_names, top_k=15):
    weights = None
    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        weights = np.abs(estimator.coef_[0])
    if weights is None:
        return []
    top_indices = np.argsort(weights)[::-1][:top_k]
    return [
        {"feature": str(feature_names[idx]), "importance": round(float(weights[idx]), 5)}
        for idx in top_indices
        if weights[idx] > 0
    ]


def run_full_pipeline():
    print("[1/6] Loading and preparing dataset...")
    df_raw = load_raw_dataset()
    df_clean, prep_summary = inspect_and_prepare_data(df_raw)
    print(f"      Cleaned rows: {prep_summary['cleaned_rows']} (Removed {prep_summary['duplicate_rows_removed']} duplicates)")

    print("[2/6] Generating EDA summaries & plots...")
    eda_summary = generate_eda_visualizations(df_clean, prep_summary)

    print("[3/6] 60/20/20 Stratified Split & v1.0 Baseline Benchmark...")
    X = df_clean["message"].values
    y = df_clean["target"].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    # v1.0 Baseline (Word TF-IDF only, 0.50 cutoff, Random Forest)
    v1_vec = TfidfVectorizer(preprocessor=preprocess_sms_text, stop_words=CUSTOM_STOP_WORDS, max_features=1500, ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    X_tr_v1 = v1_vec.fit_transform(X_train)
    X_te_v1 = v1_vec.transform(X_test)
    v1_rf = RandomForestClassifier(n_estimators=150, min_samples_split=4, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    v1_rf.fit(X_tr_v1, y_train)
    cce4_baseline_metrics = evaluate_from_scores(y_test, v1_rf.predict_proba(X_te_v1)[:, 1], threshold=0.50)

    print("[4/6] Fitting v2.0 Hybrid TF-IDF (Word + Character N-Grams) & Running Unit III Unsupervised PCA + K-Means...")
    word_tfidf = TfidfVectorizer(preprocessor=preprocess_sms_text, stop_words=CUSTOM_STOP_WORDS, max_features=1500, ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1200, min_df=3, sublinear_tf=True, dtype=np.float32)

    word_tfidf.fit(X_train)
    char_tfidf.fit(X_train)

    feature_bundle = {"word_tfidf": word_tfidf, "char_tfidf": char_tfidf}
    joblib.dump(feature_bundle, MODELS_DIR / "feature_pipeline.joblib")

    X_train_hyb = transform_hybrid_tfidf(X_train, feature_bundle)
    X_val_hyb = transform_hybrid_tfidf(X_val, feature_bundle)
    X_test_hyb = transform_hybrid_tfidf(X_test, feature_bundle)

    all_feature_names = list(word_tfidf.get_feature_names_out()) + [f"char:{c}" for c in char_tfidf.get_feature_names_out()]

    # Unsupervised Learning (PCA + K-Means Elbow)
    unsupervised_summary = run_unsupervised_analysis(df_clean, word_tfidf)

    print("[5/6] Training Unit II & IV Models + Soft-Voting Meta-Ensemble...")
    train_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    models_spec = build_models()
    model_results = {}
    comparison_rows = []
    val_prob_cache = {}
    test_prob_cache = {}
    trained_estimators = {}

    for model_id, info in models_spec.items():
        print(f"      -> Training {info['name']}...")
        estimator = info["estimator"]
        if model_id in {"adaboost", "gradient_boosting", "xgboost"}:
            estimator.fit(X_train_hyb, y_train, sample_weight=train_sample_weights)
        else:
            estimator.fit(X_train_hyb, y_train)

        joblib.dump(estimator, MODELS_DIR / f"{model_id}.joblib")
        trained_estimators[model_id] = estimator

        val_scores = estimator.predict_proba(X_val_hyb)[:, 1]
        test_scores = estimator.predict_proba(X_test_hyb)[:, 1]
        val_prob_cache[model_id] = val_scores
        test_prob_cache[model_id] = test_scores

        opt_thresh = find_optimal_threshold(y_val, val_scores)
        val_metrics = evaluate_from_scores(y_val, val_scores, threshold=opt_thresh)
        test_metrics = evaluate_from_scores(y_test, test_scores, threshold=opt_thresh)

        model_results[model_id] = {
            "id": model_id,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "optimal_threshold": opt_thresh,
            "validation": val_metrics,
            "test": test_metrics,
            "top_features": extract_feature_importances(estimator, all_feature_names, top_k=12),
        }
        comparison_rows.append({
            "model_id": model_id,
            "model_name": info["name"],
            "model_type": info["type"],
            "optimal_threshold": opt_thresh,
            "val_f1": val_metrics["f1_score"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
        })

    # v2.0 Soft-Voting Meta-Ensemble (SVM + Logistic Regression + Random Forest + XGBoost)
    print("      -> Building Soft-Voting Meta-Ensemble (SVM + LR + RF + XGBoost)...")
    voting_config = {
        "type": "soft_voting",
        "member_ids": ["svm", "logistic_regression", "random_forest", "xgboost"],
        "weights": [0.35, 0.25, 0.20, 0.20],
    }
    joblib.dump(voting_config, MODELS_DIR / "soft_voting.joblib")

    val_vote_scores = (
        0.35 * val_prob_cache["svm"]
        + 0.25 * val_prob_cache["logistic_regression"]
        + 0.20 * val_prob_cache["random_forest"]
        + 0.20 * val_prob_cache["xgboost"]
    )
    test_vote_scores = (
        0.35 * test_prob_cache["svm"]
        + 0.25 * test_prob_cache["logistic_regression"]
        + 0.20 * test_prob_cache["random_forest"]
        + 0.20 * test_prob_cache["xgboost"]
    )

    vote_thresh = find_optimal_threshold(y_val, val_vote_scores)
    vote_val_metrics = evaluate_from_scores(y_val, val_vote_scores, threshold=vote_thresh)
    vote_test_metrics = evaluate_from_scores(y_test, test_vote_scores, threshold=vote_thresh)

    model_results["soft_voting"] = {
        "id": "soft_voting",
        "name": "Soft-Voting Meta-Ensemble",
        "type": "Hybrid Meta-Ensemble",
        "description": "Combines Linear SVM (35%), Logistic Regression (25%), Random Forest (20%), and XGBoost (20%) with validation-tuned threshold.",
        "optimal_threshold": vote_thresh,
        "validation": vote_val_metrics,
        "test": vote_test_metrics,
        "top_features": model_results["random_forest"]["top_features"],
    }
    comparison_rows.append({
        "model_id": "soft_voting",
        "model_name": "Soft-Voting Meta-Ensemble",
        "model_type": "Hybrid Meta-Ensemble",
        "optimal_threshold": vote_thresh,
        "val_f1": vote_val_metrics["f1_score"],
        "test_accuracy": vote_test_metrics["accuracy"],
        "test_precision": vote_test_metrics["precision"],
        "test_recall": vote_test_metrics["recall"],
        "test_f1": vote_test_metrics["f1_score"],
        "test_roc_auc": vote_test_metrics["roc_auc"],
        "test_pr_auc": vote_test_metrics["pr_auc"],
    })

    # Select Champion Model via Validation Spam F1-score
    default_model_id = max(
        model_results.keys(),
        key=lambda m: (model_results[m]["validation"]["f1_score"], model_results[m]["validation"]["pr_auc"]),
    )
    default_model_name = model_results[default_model_id]["name"]
    winner_test = model_results[default_model_id]["test"]

    print("[6/6] Pre-training Unit V Q-Learning Adaptive Screening Agent...")
    champion_val_scores = val_vote_scores if default_model_id == "soft_voting" else val_prob_cache[default_model_id]
    q_agent_data = train_initial_q_table(y_val, champion_val_scores)
    with open(MODELS_DIR / "q_table.json", "w", encoding="utf-8") as f:
        json.dump(q_agent_data, f, indent=2)

    print(f"\n[v1.0 Baseline RF] Test Acc: {cce4_baseline_metrics['accuracy']*100:.2f}% | Spam F1: {cce4_baseline_metrics['f1_score']:.4f} | PR-AUC: {cce4_baseline_metrics['pr_auc']:.4f}")
    print(f"[v2.0 Enhanced: {default_model_name}] Test Acc: {winner_test['accuracy']*100:.2f}% | Spam F1: {winner_test['f1_score']:.4f} | PR-AUC: {winner_test['pr_auc']:.4f} | Cutoff: {winner_test['threshold']}")

    pd.DataFrame(comparison_rows).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    metrics_payload = {
        "default_model_id": default_model_id,
        "default_model_name": default_model_name,
        "selection_criterion": "Highest Validation Set Spam-Class F1-Score (with Threshold Tuning)",
        "models": model_results,
        "comparison_table": comparison_rows,
        "unsupervised_unit3": unsupervised_summary,
        "cce5_upgrade_benchmark": {
            "cce4_baseline": {
                "label": "v1.0 Baseline (Word TF-IDF + 0.50 Cutoff, Random Forest)",
                "accuracy": cce4_baseline_metrics["accuracy"],
                "precision": cce4_baseline_metrics["precision"],
                "recall": cce4_baseline_metrics["recall"],
                "f1_score": cce4_baseline_metrics["f1_score"],
                "roc_auc": cce4_baseline_metrics["roc_auc"],
                "pr_auc": cce4_baseline_metrics["pr_auc"],
                "false_positives": cce4_baseline_metrics["confusion_matrix"][0][1],
                "false_negatives": cce4_baseline_metrics["confusion_matrix"][1][0],
            },
            "cce5_enhanced": {
                "label": f"v2.0 Enhanced ({default_model_name} + Hybrid TF-IDF + Tuned Cutoff)",
                "accuracy": winner_test["accuracy"],
                "precision": winner_test["precision"],
                "recall": winner_test["recall"],
                "f1_score": winner_test["f1_score"],
                "roc_auc": winner_test["roc_auc"],
                "pr_auc": winner_test["pr_auc"],
                "false_positives": winner_test["confusion_matrix"][0][1],
                "false_negatives": winner_test["confusion_matrix"][1][0],
            },
            "feature_counts": {
                "word_tfidf": int(len(word_tfidf.get_feature_names_out())),
                "char_tfidf": int(len(char_tfidf.get_feature_names_out())),
                "total_features": len(all_feature_names),
            },
        },
    }

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("[DONE] v2.0 syllabus-aligned models, PCA, K-Means, Q-Table, and benchmarks saved!")
    return eda_summary, metrics_payload


if __name__ == "__main__":
    run_full_pipeline()
