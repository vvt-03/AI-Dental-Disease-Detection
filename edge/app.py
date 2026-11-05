
import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Dental Edge AI",
    page_icon="🦷",
    layout="centered",
)

st.title("🦷 Dental Edge AI")
st.caption("Research prototype — dental image analysis")

st.warning(
    "Research prototype only. The trained model is not connected yet; "
    "this app does not currently diagnose dental disease."
)

uploaded_file = st.file_uploader(
    "Upload a dental image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    if st.button("Analyze image", type="primary"):
        with st.spinner("Sending image to the API..."):
            try:
                response = httpx.post(
                    f"{API_URL}/analyze",
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type,
                        )
                    },
                    timeout=30.0,
                )

                if response.is_success:
                    result = response.json()
                    st.subheader("API response")
                    st.json(result)

                    if result["status"] == "model_not_connected":
                        st.info(
                            "Upload successful. Real model inference "
                            "will be enabled when the trained model is ready."
                        )
                else:
                    st.error(
                        f"API returned {response.status_code}: "
                        f"{response.text}"
                    )

            except httpx.RequestError:
                st.error(
                    "Could not reach FastAPI. Start the backend and "
                    "try again."
                )
