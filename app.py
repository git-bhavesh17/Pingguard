import json
from pathlib import Path
import joblib
from flask import Flask, jsonify, render_template, request

from train_models import (
    analyze_warning_signs,
    preprocess_sms_text,
    transform_hybrid_tfidf,
    get_rl_state,
    RL_ACTIONS,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

BASE_MODELS = [
    "decision_tree",
    "logistic_regression",
    "svm",
    "random_forest",
    "adaboost",
    "gradient_boosting",
    "xgboost",
]
EXPECTED_MODELS = BASE_MODELS + ["soft_voting"]
MAX_MESSAGE_LENGTH = 2000

ACTION_LABELS = {
    "allow_ham": "ALLOW (Deliver to Inbox)",
    "quarantine_review": "QUARANTINE (Flag for User Review)",
    "block_spam": "BLOCK (Move to Spam Folder)",
}


def create_app(model_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MODEL_DIR"] = Path(model_dir)
    app.config["REPORTS_DIR"] = Path(reports_dir)

    artifact_cache = {
        "feature_pipeline": None,
        "models": {},
        "eda_summary": None,
        "model_metrics": None,
        "q_table": None,
    }

    def load_artifacts_if_needed():
        m_dir = app.config["MODEL_DIR"]
        r_dir = app.config["REPORTS_DIR"]

        fp_path = m_dir / "feature_pipeline.joblib"
        if artifact_cache["feature_pipeline"] is None and fp_path.exists():
            artifact_cache["feature_pipeline"] = joblib.load(fp_path)

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

        q_path = m_dir / "q_table.json"
        if artifact_cache["q_table"] is None and q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                artifact_cache["q_table"] = json.load(f)

    load_artifacts_if_needed()

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        load_artifacts_if_needed()
        m_dir = app.config["MODEL_DIR"]
        missing = []
        if not (m_dir / "feature_pipeline.joblib").exists():
            missing.append("feature_pipeline.joblib")
        for m_id in EXPECTED_MODELS:
            if not (m_dir / f"{m_id}.joblib").exists():
                missing.append(f"{m_id}.joblib")

        ready = len(missing) == 0 and artifact_cache["model_metrics"] is not None
        return (
            jsonify({
                "status": "ready" if ready else "degraded",
                "artifacts_ready": ready,
                "loaded_models": list(artifact_cache["models"].keys()),
                "missing_artifacts": missing,
            }),
            200 if ready else 503,
        )

    @app.route("/api/dashboard", methods=["GET"])
    def dashboard():
        load_artifacts_if_needed()
        if artifact_cache["eda_summary"] is None or artifact_cache["model_metrics"] is None:
            return jsonify({"error": "Artifacts not found. Run `python train_models.py` first."}), 503
        return jsonify({
            "eda": artifact_cache["eda_summary"],
            "metrics": artifact_cache["model_metrics"],
            "q_table": artifact_cache["q_table"],
        })

    @app.route("/api/models", methods=["GET"])
    def list_models():
        load_artifacts_if_needed()
        if artifact_cache["model_metrics"] is None:
            return jsonify({"error": "Model metrics not available. Run `python train_models.py` first."}), 503
        metrics_data = artifact_cache["model_metrics"]
        return jsonify({
            "default_model_id": metrics_data["default_model_id"],
            "default_model_name": metrics_data["default_model_name"],
            "selection_criterion": metrics_data["selection_criterion"],
            "models": metrics_data["models"],
            "comparison_table": metrics_data["comparison_table"],
        })

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

        metrics_data = artifact_cache["model_metrics"]
        default_id = metrics_data["default_model_id"] if metrics_data else "soft_voting"
        model_id = payload.get("model_id") or default_id

        if model_id not in EXPECTED_MODELS:
            return jsonify({"error": f"Unknown model '{model_id}'."}), 400

        fp = artifact_cache["feature_pipeline"]
        model_obj = artifact_cache["models"].get(model_id)
        if fp is None or model_obj is None:
            return jsonify({"error": f"Required artifact for '{model_id}' is missing. Run `python train_models.py`."}), 503

        # Transform text using v2.0 Hybrid Word + Character N-Gram TF-IDF
        X_vec = transform_hybrid_tfidf([message], fp)

        # Compute spam probability score
        if model_id == "soft_voting" and isinstance(model_obj, dict):
            spam_score = 0.0
            for member_id, weight in zip(model_obj["member_ids"], model_obj["weights"]):
                member_est = artifact_cache["models"].get(member_id)
                if member_est is None:
                    return jsonify({"error": f"Voting member '{member_id}' is missing."}), 503
                spam_score += float(weight) * float(member_est.predict_proba(X_vec)[0, 1])
        else:
            if hasattr(model_obj, "predict_proba"):
                spam_score = float(model_obj.predict_proba(X_vec)[0, 1])
            else:
                spam_score = float(model_obj.predict(X_vec)[0])

        # Use validation-tuned threshold for this model
        threshold = 0.50
        model_display_name = model_id
        if metrics_data and model_id in metrics_data["models"]:
            threshold = float(metrics_data["models"][model_id].get("optimal_threshold", 0.50))
            model_display_name = metrics_data["models"][model_id]["name"]

        is_spam = spam_score >= threshold
        pred_label = "SPAM" if is_spam else "HAM"
        warning_signs = analyze_warning_signs(message)

        # Q-Learning Agent Policy Lookup
        rl_state = get_rl_state(spam_score)
        rl_recommendation = {
            "state": rl_state,
            "action": "block_spam" if is_spam else "allow_ham",
            "action_label": ACTION_LABELS["block_spam" if is_spam else "allow_ham"],
            "q_values": {},
        }
        if artifact_cache["q_table"] and rl_state in artifact_cache["q_table"]["q_values"]:
            state_qs = artifact_cache["q_table"]["q_values"][rl_state]
            best_action = max(state_qs.keys(), key=lambda a: state_qs[a])
            rl_recommendation = {
                "state": rl_state,
                "action": best_action,
                "action_label": ACTION_LABELS.get(best_action, best_action),
                "q_values": state_qs,
            }

        return jsonify({
            "predicted_class": pred_label,
            "is_spam": is_spam,
            "estimated_spam_score": round(spam_score, 4),
            "estimated_spam_percentage": round(spam_score * 100, 1),
            "decision_threshold": threshold,
            "model_id": model_id,
            "model_name": model_display_name,
            "warning_signs": warning_signs,
            "rl_agent": rl_recommendation,
            "disclaimer": (
                "Educational Screening Estimate: Score comes from Hybrid Word+Char TF-IDF models with validation-tuned cutoffs. "
                "A HAM result does not guarantee safety; rule-based warning signs and the Q-Learning policy act as supplementary decision aids."
            ),
        })

    @app.route("/api/rl/feedback", methods=["POST"])
    def rl_feedback():
        """Live Bellman Q-Learning update based on user feedback."""
        load_artifacts_if_needed()
        q_data = artifact_cache["q_table"]
        if q_data is None:
            return jsonify({"error": "Q-Table not initialized. Run `python train_models.py` first."}), 503

        payload = request.get_json(silent=True) or {}
        state = payload.get("state")
        action = payload.get("action")
        is_correct = payload.get("is_correct")

        if state not in q_data["q_values"] or action not in RL_ACTIONS or not isinstance(is_correct, bool):
            return jsonify({"error": "Invalid RL feedback parameters."}), 400

        reward = 10.0 if is_correct else -15.0
        alpha = float(q_data.get("alpha", 0.15))
        gamma = float(q_data.get("gamma", 0.80))

        old_q = float(q_data["q_values"][state][action])
        max_next_q = max(q_data["q_values"][state].values())
        new_q = round(old_q + alpha * (reward + gamma * max_next_q - old_q), 3)

        q_data["q_values"][state][action] = new_q
        q_data["feedback_updates"] = int(q_data.get("feedback_updates", 0)) + 1

        with open(app.config["MODEL_DIR"] / "q_table.json", "w", encoding="utf-8") as f:
            json.dump(q_data, f, indent=2)

        return jsonify({
            "message": f"Q-Learning Agent updated! State='{state}', Action='{action}', Reward={reward:+.0f} (Q: {old_q} -> {new_q})",
            "reward": reward,
            "old_q": old_q,
            "new_q": new_q,
            "q_table": q_data,
        })

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
