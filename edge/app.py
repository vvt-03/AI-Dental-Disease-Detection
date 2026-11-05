
import httpx
import streamlit as st
from PIL import Image, ImageDraw

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Dental Edge AI",
    page_icon="🦷",
    layout="centered",
)

st.title("🦷 Dental Edge AI")
st.caption("Research prototype — dental image analysis")


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
                    st.subheader("YOLO11m Detection Results")
                    st.write("Status:", result["status"])
                    st.write("Model:", result["model_architecture"])
                    st.write("Inference device:", result["device"])
                    detections = result["detections"]

                    if detections:
                        image = Image.open(uploaded_file).convert("RGB")
                        draw = ImageDraw.Draw(image)

                        for item in detections:
                            x1, y1, x2, y2 = item["bbox_xyxy"]
                            label = (
                                f'{item["class_name"]} '
                                f'{item["confidence"]:.2%}'
                            )

                            draw.rectangle(
                                [x1, y1, x2, y2],
                                outline="red",
                                width=3,
                            )

                            draw.text(
                                (x1, max(0, y1 - 18)),
                                label,
                                fill="red",
                            )

                        st.image(
                            image,
                            caption="YOLO11m detected regions",
                            use_container_width=True,
                        )

                        st.subheader("Detection Details")

                        st.dataframe(
                            [
                                {
                                    "Class": item["class_name"],
                                    "Confidence": item["confidence"],
                                    "Bounding Box": item["bbox_xyxy"],
                                }
                                for item in detections
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No objects detected above the confidence threshold.")

                else:
                    st.error(
                        f"Analysis failed (HTTP {response.status_code}): "
                        f"{response.text}"
                    )

            except httpx.RequestError:
                st.error(
                    "Could not reach FastAPI. Start the backend and "
                    "try again."
                )
