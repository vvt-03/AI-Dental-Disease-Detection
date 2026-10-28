"""Edge application entry point."""


import streamlit as st

st.set_page_config(
    page_title="Dental Edge AI",
    page_icon="🦷"
)

st.title("Dental Edge AI")
st.caption("Edge-based Dental Image Analysis")

uploaded = st.file_uploader(
    "Upload a dental X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded dental image")

    st.success("Image uploaded successfully!")
    st.info(
        "AI inference will be integrated on Day 2–3."
    )
