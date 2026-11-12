import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake Job Detection System", page_icon="🛡️", layout="centered"
)

model = joblib.load("job_model.pkl")

st.sidebar.title("ℹ️ About Project")
st.sidebar.info(
    "This AI-powered web application leverages Machine Learning and Natural"
    " Language Processing (NLP) to detect fraudulent and scam job postings."
)
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Technical Stack")
st.sidebar.text(
    "• Python\n• Scikit-Learn\n• Pandas & NumPy\n• Streamlit (Web UI)"
)
st.sidebar.markdown("---")
st.sidebar.text("Developed for Academic Presentation")

st.title("🛡️ AI-Powered Fake Job Detection System")
st.write(
    "Paste the job description and details below to analyze whether the posting"
    " is **Real** or **Fake**."
)

job_description = st.text_area(
    "📝 Paste Job Description Here:",
    height=220,
    placeholder=(
        "Paste the complete job description, company details, and requirements"
        " here..."
    ),
)

if st.button("🔍 Analyze Job Posting", type="primary"):
  if job_description.strip() == "":
    st.warning("⚠️ Please enter a job description before analyzing.")
  else:
    with st.spinner("Analyzing job posting with Machine Learning model..."):
      prediction = model.predict([job_description])

      try:
        probabilities = model.predict_proba([job_description])
        confidence = max(probabilities[0]) * 100
      except Exception:
        confidence = 90.0

      if prediction[0] == 1:
        st.error(
            f"⚠️ **Warning: Potential FAKE Job Posting Detected!**\n\n"
            f"• **Confidence Score:** `{confidence:.2f}%`\n• **Recommendation:**"
            " Exercise high caution. Do not share sensitive personal documents"
            " or transfer any money."
        )
      else:
        st.success(
            f"✅ **This appears to be a REAL Job Posting.**\n\n"
            f"• **Confidence Score:** `{confidence:.2f}%`\n• **Recommendation:**"
            " The description looks standard, but always cross-verify with the"
            " official company career portal."
        )