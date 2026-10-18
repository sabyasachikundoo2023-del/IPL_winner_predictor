import logging
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Load model and encoders once at start-up
# --------------------------------------------------------------------------
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

team_encoder = encoders["team_encoder"]
venue_encoder = encoders["venue_encoder"]
toss_decision_encoder = encoders["toss_decision_encoder"]
feature_cols = encoders["feature_cols"]

TEAMS = sorted(team_encoder.classes_.tolist())
VENUES = sorted(venue_encoder.classes_.tolist())
TOSS_DECISIONS = sorted(toss_decision_encoder.classes_.tolist())

# Map each class the model knows to its column in predict_proba output
CLASS_INDEX = {int(c): i for i, c in enumerate(model.classes_)}


def render_home(**context):
    """Render the home page with dropdown options and any extra context."""
    return render_template(
        "index.html",
        teams=TEAMS,
        venues=VENUES,
        toss_decisions=TOSS_DECISIONS,
        form=context.pop("form", {}),
        **context,
    )


@app.route("/")
def home():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    form = {
        "team1": request.form.get("team1", "").strip(),
        "team2": request.form.get("team2", "").strip(),
        "venue": request.form.get("venue", "").strip(),
        "toss_winner": request.form.get("toss_winner", "").strip(),
        "toss_decision": request.form.get("toss_decision", "").strip(),
    }

    # ---- 1. Missing inputs -------------------------------------------------
    missing = [k.replace("_", " ").title() for k, v in form.items() if not v]
    if missing:
        return render_home(error=f"Please select: {', '.join(missing)}.", form=form), 400

    # ---- 2. Logical validation --------------------------------------------
    if form["team1"] == form["team2"]:
        return render_home(error="Team 1 and Team 2 must be different.", form=form), 400

    if form["toss_winner"] not in (form["team1"], form["team2"]):
        return render_home(
            error="The toss winner must be either Team 1 or Team 2.", form=form
        ), 400

    # ---- 3. Encode with the SAME encoders used in training -----------------
    try:
        row = pd.DataFrame(
            [
                {
                    "team1": team_encoder.transform([form["team1"]])[0],
                    "team2": team_encoder.transform([form["team2"]])[0],
                    "venue": venue_encoder.transform([form["venue"]])[0],
                    "toss_winner": team_encoder.transform([form["toss_winner"]])[0],
                    "toss_decision": toss_decision_encoder.transform(
                        [form["toss_decision"]]
                    )[0],
                }
            ]
        )[feature_cols]
    except ValueError as exc:
        logger.warning("Encoding error: %s", exc)
        return render_home(
            error="One of the selected values was not seen during training. "
            "Please choose from the dropdown lists.",
            form=form,
        ), 400

    # ---- 4. Predict --------------------------------------------------------
    try:
        proba = model.predict_proba(row)[0]
        t1_code, t2_code = int(row.at[0, "team1"]), int(row.at[0, "team2"])

        # The model knows every franchise, but only two are playing: compare
        # the probabilities of those two teams only.
        p1 = proba[CLASS_INDEX[t1_code]] if t1_code in CLASS_INDEX else 0.0
        p2 = proba[CLASS_INDEX[t2_code]] if t2_code in CLASS_INDEX else 0.0
        total = p1 + p2
        if total > 0:
            p1, p2 = p1 / total, p2 / total
        else:  # model has no signal for either team
            p1 = p2 = 0.5

        winner = form["team1"] if p1 >= p2 else form["team2"]
        confidence = round(max(p1, p2) * 100, 1)
    except Exception:
        logger.exception("Prediction failed")
        return render_home(
            error="Something went wrong while making the prediction. Please try again.",
            form=form,
        ), 500

    return render_home(
        prediction=winner,
        confidence=confidence,
        probabilities={
            form["team1"]: round(p1 * 100, 1),
            form["team2"]: round(p2 * 100, 1),
        },
        form=form,
    )


if __name__ == "__main__":
    app.run(debug=True)
