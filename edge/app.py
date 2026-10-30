
import json
import streamlit as st

st.set_page_config(
    page_title="Dental Edge AI",
    page_icon="🦷",
    layout="centered",
)

st.title("🦷 Dental Edge AI")
st.caption("Research prototype — edge-based dental image analysis")

st.info(
    "Image upload and preview are ready. Model inference is not "
    "integrated yet, so this interface does not diagnose dental disease."
)

uploaded_file = st.file_uploader(
    "Upload a dental image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded image preview",
        use_container_width=True,
    )
    st.success(f"Image loaded: {uploaded_file.name}")

    st.subheader("Prediction JSON format")
    example_payload = {
        "image_id": uploaded_file.name,
        "predicted_class": "pending_inference",
        "confidence": 0.0,
        "model_version": "not-integrated",
    }
    st.code(json.dumps(example_payload, indent=2), language="json")
    st.caption(
        "This is a schema preview only, not a prediction from an AI model."
    )
