import io
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Crop Planner AI", page_icon="🌱", layout="wide")
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Crop_Field_Prediction_RF.pkl"
DATA_PATHS = [BASE_DIR / "archive (2).zip", BASE_DIR / "crop_dataset.csv", BASE_DIR / "Crop_Field_Prediction.csv"]
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
FEATURES = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL"]

st.markdown("""
<style>
.stApp {background:#f5f8f4} .block-container {padding-top:1.5rem;max-width:1250px}
.hero {padding:25px 30px;border-radius:16px;background:linear-gradient(110deg,#174b32,#2d8050);color:white;margin-bottom:20px}
.hero h1 {color:white;margin:0 0 6px} .hero p {color:#e7f4e9;margin:0}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:12px;border:1px solid #e1e9df}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌱 Smart Crop Planner AI</h1><p>Explore crop data, review model results, and predict a crop from soil and weather measurements.</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_saved_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

def unpack_model(saved):
    if isinstance(saved, dict):
        return (saved["model"], saved.get("scaler"), saved.get("needs_scaling", False),
                list(saved.get("features_cols", FEATURES)))
    return saved, None, False, FEATURES

def make_pdf(report):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Smart Crop Planner AI - Prediction Report", styles["Title"]), Spacer(1, 12),
             Paragraph(f"Generated: {report['timestamp']}", styles["Normal"]), Spacer(1, 12),
             Paragraph(f"Predicted crop: <b>{report['prediction']}</b>", styles["Heading2"]), Spacer(1, 10)]
    rows = [["Input", "Value"]] + [[key, str(value)] for key, value in report["inputs"].items()]
    table = Table(rows, colWidths=[220, 220])
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#174b32")),
                               ("TEXTCOLOR", (0,0),(-1,0),colors.white),
                               ("GRID", (0,0),(-1,-1),0.5,colors.HexColor("#d5dfd2")),
                               ("PADDING", (0,0),(-1,-1),8)]))
    story.append(table)
    if report.get("score") is not None:
        story += [Spacer(1, 12), Paragraph(f"Highest model probability: {report['score']:.1f}% (estimate, not a guarantee).", styles["Normal"])]
    story += [Spacer(1, 18), Paragraph("Use reliable local agricultural guidance before making real farming decisions.", styles["Italic"])]
    doc.build(story)
    return buffer.getvalue()

def load_history():
    """Load previously saved predictions from CSV."""
    if not HISTORY_PATH.exists():
        return []
    try:
        saved = pd.read_csv(HISTORY_PATH)
        history = []
        for _, row in saved.iterrows():
            inputs = {feature: row[feature] for feature in FEATURES if feature in saved.columns}
            score_value = row.get("score", None)
            score = None if pd.isna(score_value) else float(score_value)
            history.append({
                "timestamp": str(row["timestamp"]),
                "prediction": str(row["prediction"]),
                "inputs": inputs,
                "score": score,
            })
        return history
    except Exception as exc:
        st.warning(f"Could not load saved prediction history: {exc}")
        return []


def save_history(history):
    """Save prediction history so it remains available after restarting the app."""
    rows = []
    for item in history:
        row = {
            "timestamp": item["timestamp"],
            "prediction": item["prediction"],
            "score": item["score"],
        }
        row.update(item["inputs"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(HISTORY_PATH, index=False)


if "history" not in st.session_state:
    st.session_state.history = load_history()
if "last_report" not in st.session_state:
    st.session_state.last_report = (
        st.session_state.history[0] if st.session_state.history else None
    )

model = scaler = None
needs_scaling = False
features = FEATURES
model_error = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Put {MODEL_PATH.name} in the same folder as app.py.")
    model, scaler, needs_scaling, features = unpack_model(load_saved_model())
except Exception as exc:
    model_error = str(exc)

with st.sidebar:
    st.header("🌿 Project Menu")
    st.caption("Smart Crop Planner AI")
    if model_error:
        st.error("Model not ready")
        st.caption(model_error)
    else:
        st.success("Saved model loaded")
        st.caption(f"Loaded model: {type(model).__name__}")
    st.caption("Educational project; predictions are estimates.")

tab_predict, tab_data, tab_models, tab_info, tab_history = st.tabs([
    "🌾 Crop Prediction", "📊 Dataset Analytics", "🧠 Model Comparison", "🌿 Crop Information", "🕘 History"])

with tab_predict:
    st.subheader("Enter field conditions")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            n = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
            p = st.number_input("Phosphorus (P)", min_value=0.0, value=40.0, step=1.0)
            k = st.number_input("Potassium (K)", min_value=0.0, value=40.0, step=1.0)
            temp = st.number_input("Temperature (°C)", value=25.0, step=0.5)
        with right:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0, step=1.0)
        predict = st.form_submit_button("🌱 Predict Suitable Crop", use_container_width=True)

    if predict:
        if model_error:
            st.error("Model loading failed. Check the error in the sidebar.")
        else:
            try:
                raw = {"N_SOIL": n, "P_SOIL": p, "K_SOIL": k, "TEMPERATURE": temp,
                       "HUMIDITY": humidity, "ph": ph, "RAINFALL": rainfall}
                input_df = pd.DataFrame([raw])[features]
                if needs_scaling:
                    if scaler is None:
                        raise ValueError("This saved model requires a scaler, but none was saved.")
                    model_input = scaler.transform(input_df)
                else:
                    model_input = input_df
                prediction = str(model.predict(model_input)[0])
                score = float(max(model.predict_proba(model_input)[0]) * 100) if hasattr(model, "predict_proba") else None
                report = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "prediction": prediction,
                          "inputs": raw, "score": score}
                st.session_state.last_report = report
                st.session_state.history.insert(0, report)
                st.session_state.history = st.session_state.history[:100]
                save_history(st.session_state.history)
                st.success(f"Predicted crop: {prediction}")
                a, b, c = st.columns(3)
                a.metric("Predicted crop", prediction)
                b.metric("Input features", len(raw))
                c.metric("Top model score", f"{score:.1f}%" if score is not None else "Not available")
                st.caption("Model scores are estimates, not guarantees of real-world crop suitability.")
                if hasattr(model, "predict_proba"):
                    pairs = sorted(zip(model.classes_, model.predict_proba(model_input)[0]), key=lambda x: x[1], reverse=True)[:3]
                    chart = pd.DataFrame(pairs, columns=["Crop", "Probability"]).set_index("Crop") * 100
                    st.subheader("Top 3 model predictions")
                    st.bar_chart(chart)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check that the saved model uses these seven features and the same preprocessing as training.")

    if st.session_state.last_report:
        try:
            st.download_button("📄 Download prediction report (PDF)", data=make_pdf(st.session_state.last_report),
                               file_name="crop_prediction_report.pdf", mime="application/pdf")
        except ImportError:
            st.warning("Install PDF support with: python -m pip install reportlab")

with tab_data:
    st.subheader("Dataset Analytics")
    data_path = next((path for path in DATA_PATHS if path.exists()), None)
    if data_path is None:
        st.info("To enable analytics, copy archive (2).zip or a CSV dataset into the same folder as app.py.")
    else:
        try:
            data = load_dataset(str(data_path))
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(data):,}")
            b.metric("Columns", len(data.columns))
            c.metric("Missing values", int(data.isna().sum().sum()))
            st.subheader("Dataset preview")
            st.dataframe(data.head(10), use_container_width=True)
            if "CROP" in data:
                st.subheader("Records by crop")
                st.bar_chart(data["CROP"].value_counts().sort_values(ascending=False))
            if "CROP_PRICE" in data and "CROP" in data:
                st.subheader("Average crop price in dataset")
                st.bar_chart(data.groupby("CROP")["CROP_PRICE"].mean().sort_values(ascending=False))
        except Exception as exc:
            st.warning(f"Could not read dataset: {exc}. A ZIP should contain a readable CSV.")

with tab_models:
    st.subheader("Model Comparison")
    if COMPARISON_PATH.exists():
        try:
            comparison = pd.read_csv(COMPARISON_PATH)
            st.dataframe(comparison, use_container_width=True)
            if {"Model", "Accuracy"}.issubset(comparison.columns):
                st.bar_chart(comparison.set_index("Model")[["Accuracy"]])
                best = comparison.loc[comparison["Accuracy"].idxmax()]
                st.success(f"Highest listed accuracy: {best['Model']} ({best['Accuracy']:.3f})")
            else:
                st.warning("CSV must have columns named Model and Accuracy.")
        except Exception as exc:
            st.warning(f"Could not read model_comparison.csv: {exc}")
    else:
        st.info("Export your notebook's model comparison into model_comparison.csv to show the chart here.")
        st.code('comparison_df = pd.DataFrame(results, columns=["Model", "Accuracy"])\ncomparison_df.to_csv("model_comparison.csv", index=False)', language="python")
        st.caption("Run this in the notebook after your comparison code creates the results variable.")
    if model is not None and hasattr(model, "feature_importances_"):
        st.subheader("Feature importance for the loaded model")
        st.bar_chart(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False))

with tab_info:
    st.subheader("Crop Information")
    descriptions = {
        "Apple": "A fruit tree crop; variety choice and climate suitability depend on the region.",
        "Banana": "A tropical fruit crop with variety-specific water and nutrient needs.",
        "Blackgram": "A pulse crop; season and management depend on local conditions.",
        "ChickPea": "A pulse crop; soil drainage, season, and variety matter.",
        "Coconut": "A perennial tropical crop; long-term climate and soil are important.",
        "Coffee": "A perennial crop whose variety, elevation, and rainfall affect suitability.",
        "Cotton": "A fibre crop; season, water needs, and pest management vary by region.",
        "Grapes": "A fruit crop requiring variety- and region-specific management.",
        "Jute": "A fibre crop commonly associated with warm, humid conditions in suitable regions.",
        "KidneyBeans": "A pulse crop; local climate and cultivar influence growing requirements.",
        "Lentil": "A pulse crop often grown in cooler seasons in suitable regions.",
        "Maize": "A cereal crop used for food and feed; conditions vary by variety and location.",
        "Mango": "A fruit tree crop whose variety and management depend on the region.",
        "MothBeans": "A pulse crop grown in some warm, dry regions; local advice is important.",
        "MungBean": "A pulse crop; sowing time and conditions depend on local farming systems.",
        "Muskmelon": "A fruit crop; temperature, soil drainage, and variety matter.",
        "Orange": "A citrus fruit crop; rootstock, soil, and climate affect suitability.",
        "Papaya": "A tropical fruit crop sensitive to temperature and soil drainage.",
        "PigeonPeas": "A pulse crop used for food; duration and water needs vary by variety.",
        "Pomegranate": "A fruit crop; irrigation and soil management depend on the region.",
        "Rice": "A cereal crop commonly grown in warm conditions; water needs vary by system and variety.",
        "Watermelon": "A fruit crop; temperature, soil drainage, and variety matter.",
    }
    selected = st.selectbox("Select a crop", sorted(descriptions))
    st.info(descriptions[selected])
    st.caption("These are general descriptions, not personalized cultivation instructions. Verify local farming decisions with an agricultural expert or official regional guidance.")

with tab_history:
    st.subheader("Prediction History")
    if st.session_state.history:
        rows = []
        for item in st.session_state.history:
            row = {"Timestamp": item["timestamp"], "Predicted crop": item["prediction"],
                   "Top model score (%)": round(item["score"], 2) if item["score"] is not None else "N/A"}
            row.update(item["inputs"])
            rows.append(row)
        history_df = pd.DataFrame(rows)
        st.dataframe(history_df, use_container_width=True)
        st.download_button("⬇️ Download history as CSV", history_df.to_csv(index=False).encode("utf-8"),
                           file_name="crop_prediction_history.csv", mime="text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last_report = None
            if HISTORY_PATH.exists():
                HISTORY_PATH.unlink()
            st.rerun()
        st.caption("History is saved to prediction_history.csv and remains available after restarting the app.")
    else:
        st.info("No predictions yet. Make a prediction in the Crop Prediction tab to build history.")

st.markdown("---")
st.caption("Smart Crop Planner AI • Educational project • Verify farming decisions with reliable local advice.")