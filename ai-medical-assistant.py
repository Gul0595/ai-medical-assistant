import streamlit as st
from medical_logic import (
    generate_medical_response,
    assess_severity,
    format_medical_response
)

st.set_page_config(page_title="AI Medical Assistant")

st.title("🩺 AI Medical Knowledge Assistant")

st.warning("⚠️ Educational use only. Not medical advice.")

symptoms = st.text_area(
    "Describe your symptoms:",
    placeholder="fever, headache, body pain for two days"
)

if st.button("Get Educational Insight"):
    if not symptoms.strip():
        st.error("Please enter symptoms.")
    else:
        with st.spinner("Analyzing..."):
            raw = generate_medical_response(symptoms)
            severity, msg = assess_severity(symptoms)
            final = format_medical_response(raw, severity, msg)

        if severity == "HIGH":
            st.error(msg)
        elif severity == "MODERATE":
            st.warning(msg)
        else:
            st.success(msg)

        st.markdown(final)