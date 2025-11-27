import json
import os

import requests
import streamlit as st
from PIL import Image

# Default API URL (FastAPI)
DEFAULT_API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Traffic Detection UI",
    page_icon="🚦",
    layout="centered",
)

st.title("🚦 Traffic Detection – Model")

st.markdown(
    "Upload an image and the app will send it to the FastAPI endpoint "
    "for YOLOv8-based traffic detection."
)

# --- API URL settings ---
with st.expander("API Settings", expanded=False):
    api_base = st.text_input(
        "FastAPI base URL",
        value=os.getenv("API_URL", DEFAULT_API_URL),
        help="Base URL of your FastAPI server (without trailing slash).",
    )

api_predict_url = f"{api_base.rstrip('/')}/predict-image"

st.write(f"Using prediction endpoint: `{api_predict_url}`")

# --- Image upload ---
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Choose a traffic-related image for detection.",
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Sending image to API and running detection..."):
            try:
                # Prepare multipart/form-data payload
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(api_predict_url, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Detection completed successfully!")

                    # --- Summary instead of raw JSON only ---
                    detections = result.get("detections", [])
                    if not detections:
                        st.warning("No objects detected in this image.")
                    else:
                        from collections import Counter

                        class_names = [det["class_name"] for det in detections]
                        counts = Counter(class_names)

                        parts = [
                            f"{count} {name}{'' if count == 1 else 's'}"
                            for name, count in counts.items()
                        ]
                        summary = ", ".join(parts)

                        st.subheader("Detection Summary")
                        st.write(f"Detected: **{summary}**")

                        # Optional detailed view
                        with st.expander("Show raw JSON (detailed results)", expanded=False):
                            st.code(json.dumps(result, indent=2), language="json")

                else:
                    st.error(
                        f"API returned status code {response.status_code}.\n"
                        f"Details: {response.text}"
                    )
            except requests.exceptions.RequestException as e:
                st.error(f"Request to API failed: {e}")
else:
    st.info("Please upload an image to get started.")