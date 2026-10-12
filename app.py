
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ---------------- LOAD SAVED FILES ----------------
@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_DIR / "spam_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
    metrics = joblib.load(MODEL_DIR / "metrics.pkl")
    return model, vectorizer, metrics

try:
    model, vectorizer, metrics = load_model_files()
except FileNotFoundError:
    st.error("Model files not found. Run: python train_model.py")
    st.stop()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📧 SpamGuard AI")
st.sidebar.caption("Machine Learning Email Protection")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Email Checker", "Prediction History", "About"]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Email Spam Classifier Dashboard")
    st.write(
        "Monitor model performance and explore the dataset "
        "used to train the spam detection system."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Messages", f"{metrics['total_messages']:,}")
    col2.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col3.metric("Spam Messages", f"{metrics['spam_count']:,}")
    col4.metric("Legitimate Messages", f"{metrics['ham_count']:,}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Message Distribution")

        fig, ax = plt.subplots()
        ax.pie(
            [metrics["ham_count"], metrics["spam_count"]],
            labels=["Legitimate", "Spam"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Spam Detection Performance")

        performance = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-score"],
            "Score": [
                metrics["precision"],
                metrics["recall"],
                metrics["f1"]
            ]
        })

        st.bar_chart(performance.set_index("Metric"))

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Legitimate", "Actual Spam"],
        columns=["Predicted Legitimate", "Predicted Spam"]
    )

    st.dataframe(cm, use_container_width=True)

# ---------------- EMAIL CHECKER ----------------
elif page == "Email Checker":
    st.title("✉️ Email Spam Checker")
    st.write("Enter a message below to predict whether it is spam.")

    message = st.text_area(
        "Email or message content",
        placeholder="Example: Congratulations! You have won a prize...",
        height=180
    )

    if st.button("Check Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            features = vectorizer.transform([message])
            prediction = model.predict(features)[0]

            # LinearSVC decision score is not a probability.
            score = float(model.decision_function(features)[0])

            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": message,
                "prediction": prediction
            })

            if prediction == "spam":
                st.error("🚨 This message is classified as SPAM.")
            else:
                st.success("✅ This message is classified as LEGITIMATE (HAM).")

            st.caption(
                f"Model decision score: {score:.3f} "
                "(not a probability or confidence percentage)"
            )

# ---------------- PREDICTION HISTORY ----------------
elif page == "Prediction History":
    st.title("🕘 Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History as CSV",
            data=csv,
            file_name="spam_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No messages have been checked in this session yet.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("About This Project")

    st.write("""
    **Email Spam Classifier** is a machine-learning project that
    classifies text messages as spam or legitimate.

    **Technologies**
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF text vectorization
    - Linear Support Vector Machine (SVM)
    - Streamlit
    - Matplotlib

    **Dataset:** UCI SMS Spam Collection

    **Important:** This prototype classifies message text. It does not
    connect to an inbox, verify the sender, or scan email attachments.
    """)

st.sidebar.divider()
st.sidebar.caption("Email Spam Classifier | Internship Project")
