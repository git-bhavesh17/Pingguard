import json
from pathlib import Path
import joblib
from flask import Flask, jsonify, render_template, request

from train_models import analyze_warning_signs, preprocess_sms_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

EXPECTED_MODELS = [
    "decision_tree",
    "random_forest",
    "adaboost",
    "gradient_boosting",
    "xgboost",
]
MAX_MESSAGE_LENGTH = 2000


def create_app(model_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MODEL_DIR"] = Path(model_dir)
    app.config["REPORTS_DIR"] = Path(reports_dir)

    artifact_cache = {
        "vectorizer": None,
        "models": {},
        "eda_summary": None,
        "model_metrics": None,
    }

    def load_artifacts_if_needed():
        m_dir = app.config["MODEL_DIR"]
        r_dir = app.config["REPORTS_DIR"]

        vec_path = m_dir / "tfidf_vectorizer.joblib"
        if artifact_cache["vectorizer"] is None and vec_path.exists():
            artifact_cache["vectorizer"] = joblib.load(vec_path)

        for m_id in EXPECTED_MODELS:
            m_path = m_dir / f"{m_id}.joblib"
            if m_id not in artifact_cache["models"] and m_path.exists():
                artifact_cache["models"][m_id] = joblib.load(m_path)

        eda_path = r_dir / "eda_summary.json"
        if artifact_cache["eda_summary"] is None and eda_path.exists():
            with open(eda_path, "r", encoding="utf-8") as f:
                artifact_cache["eda_summary"] = json.load(f)

        metrics_path = r_dir / "model_metrics.json"
        if artifact_cache["model_metrics"] is None and metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                artifact_cache["model_metrics"] = json.load(f)

    load_artifacts_if_needed()

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        load_artifacts_if_needed()
        m_dir = app.config["MODEL_DIR"]
        missing = []
        if not (m_dir / "tfidf_vectorizer.joblib").exists():
            missing.append("tfidf_vectorizer.joblib")
        for m_id in EXPECTED_MODELS:
            if not (m_dir / f"{m_id}.joblib").exists():
                missing.append(f"{m_id}.joblib")

        ready = len(missing) == 0 and artifact_cache["model_metrics"] is not None
        return (
            jsonify(
                {
                    "status": "ready" if ready else "degraded",
                    "artifacts_ready": ready,
                    "loaded_models": list(artifact_cache["models"].keys()),
                    "missing_artifacts": missing,
                }
            ),
            200 if ready else 503,
        )

    @app.route("/api/dashboard", methods=["GET"])
    def dashboard():
        load_artifacts_if_needed()
        if artifact_cache["eda_summary"] is None or artifact_cache["model_metrics"] is None:
            return jsonify({"error": "Artifacts not found. Run `python train_models.py` first."}), 503
        return jsonify({"eda": artifact_cache["eda_summary"], "metrics": artifact_cache["model_metrics"]})

    @app.route("/api/models", methods=["GET"])
    def list_models():
        load_artifacts_if_needed()
        if artifact_cache["model_metrics"] is None:
            return jsonify({"error": "Model metrics not available. Run `python train_models.py` first."}), 503
        metrics_data = artifact_cache["model_metrics"]
        return jsonify(
            {
                "default_model_id": metrics_data["default_model_id"],
                "default_model_name": metrics_data["default_model_name"],
                "selection_criterion": metrics_data["selection_criterion"],
                "models": metrics_data["models"],
                "comparison_table": metrics_data["comparison_table"],
            }
        )

    @app.route("/api/predict", methods=["POST"])
    def predict():
        load_artifacts_if_needed()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload."}), 400

        message = payload.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Please enter a non-empty SMS message to analyze."}), 400

        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({"error": f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters."}), 400

        default_id = (
            artifact_cache["model_metrics"]["default_model_id"]
            if artifact_cache["model_metrics"]
            else "random_forest"
        )
        model_id = payload.get("model_id") or default_id

        if model_id not in EXPECTED_MODELS:
            return jsonify({"error": f"Unknown model '{model_id}'."}), 400

        vectorizer = artifact_cache["vectorizer"]
        model = artifact_cache["models"].get(model_id)

        if vectorizer is None or model is None:
            return jsonify({"error": f"Model artifact '{model_id}' is missing. Run `python train_models.py`."}), 503

        X_vec = vectorizer.transform([message])
        pred_num = int(model.predict(X_vec)[0])
        pred_label = "SPAM" if pred_num == 1 else "HAM"

        if hasattr(model, "predict_proba"):
            spam_score = float(model.predict_proba(X_vec)[0, 1])
        else:
            spam_score = float(pred_num)

        warning_signs = analyze_warning_signs(message)
        model_display_name = (
            artifact_cache["model_metrics"]["models"][model_id]["name"]
            if artifact_cache["model_metrics"] and model_id in artifact_cache["model_metrics"]["models"]
            else model_id
        )

        return jsonify(
            {
                "predicted_class": pred_label,
                "is_spam": pred_label == "SPAM",
                "estimated_spam_score": round(spam_score, 4),
                "estimated_spam_percentage": round(spam_score * 100, 1),
                "model_id": model_id,
                "model_name": model_display_name,
                "warning_signs": warning_signs,
                "disclaimer": (
                    "Educational Screening Estimate: This score comes from a model trained on historical English SMS data. "
                    "A HAM prediction does not guarantee safety, and rule-based warning signs are supplementary clues separate from the ML classifier."
                ),
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)