import streamlit as st
import requests
from PIL import Image
import io
import json
import logging
from typing import Optional, Tuple

# ----------------------------
# Configuration / Constants
# ----------------------------
DEFAULT_API_BASE_URL = "http://localhost:8222/api/v1"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
REQUEST_TIMEOUT = 60  # seconds

# Streamlit page config
st.set_page_config(page_title="Palm Oil Disease Detection", layout="wide")

# Lightweight logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("palm_app")


# ----------------------------
# Helper functions
# ----------------------------

def bytes_to_kb(n: int) -> float:
    return n / 1024.0


def is_valid_image_file(file) -> Tuple[bool, Optional[str]]:
    """Basic validation for uploaded file size and type."""
    if file is None:
        return False, "No file provided."

    if hasattr(file, "size") and file.size > MAX_FILE_SIZE:
        return False, f"File too large: {bytes_to_kb(file.size):.0f} KB (max {bytes_to_kb(MAX_FILE_SIZE):.0f} KB)"

    try:
        # Try opening with PIL to ensure it's an image
        file.seek(0)
        Image.open(io.BytesIO(file.read()))
        file.seek(0)
    except Exception:
        return False, "Uploaded file is not a supported image or is corrupted."

    return True, None


@st.cache_data(show_spinner=False)
def test_api_health(api_base_url: str) -> Tuple[bool, Optional[str]]:
    """Check API health and return (is_up, message_or_version). Cached to avoid repeated pings."""
    try:
        resp = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=5)
        if resp.status_code == 200:
            # If the API returns a json payload with a version field, show it
            try:
                data = resp.json()
                version = data.get("version") or data.get("api_version")
                return True, version
            except Exception:
                return True, None
        else:
            return False, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def post_file(endpoint: str, file, timeout: int = REQUEST_TIMEOUT) -> Tuple[Optional[dict], Optional[str]]:
    """Helper: POST a file to the given endpoint and return parsed JSON or error string."""
    try:
        files = {"file": ("image.jpg", file, "application/octet-stream")}
        with requests.Session() as s:
            resp = s.post(endpoint, files=files, timeout=timeout)
        try:
            payload = resp.json()
        except Exception:
            payload = None

        if resp.status_code == 200:
            return payload, None
        else:
            # Try to extract a helpful message
            if payload and isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("message") or json.dumps(payload)
            else:
                detail = resp.text
            return None, f"{resp.status_code}: {detail}"

    except requests.exceptions.Timeout:
        return None, "Request timed out. The backend may be busy or the file is large."
    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to API. Verify API URL and network connectivity."
    except Exception as e:
        logger.exception("Unexpected error while posting file")
        return None, str(e)


# ----------------------------
# UI: Sidebar
# ----------------------------

st.sidebar.header("Application Settings")
api_base_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
if st.sidebar.button("Test API Connection"):
    with st.spinner("Pinging API..."):
        is_up, info = test_api_health(api_base_url)
    if is_up:
        if info:
            st.sidebar.success(f"API reachable — version: {info}")
        else:
            st.sidebar.success("API reachable")
    else:
        st.sidebar.error(f"API unreachable — {info}")

st.sidebar.markdown("---")
st.sidebar.subheader("Detectable Conditions")
disease_classes = [
    "Black Scorch",
    "Fusarium Wilt",
    "Healthy sample",
    "Leaf Spots",
    "Magnesium Deficiency",
    "Manganese Deficiency",
    "Parlatoria Blanchardi",
    "Potassium Deficiency",
    "Rachis Blight",
]
for d in disease_classes:
    st.sidebar.text(f"• {d}")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If your backend runs on a different machine, update the API Base URL above.")


# ----------------------------
# Main area
# ----------------------------

st.title("Palm Oil Disease Detection")
st.markdown(
    "Upload a clear photo of a palm oil leaf. Choose Quick Prediction (fast) or Full Analysis (detailed report + recommendations)."
)

# File upload area
uploaded_file = st.file_uploader("Upload image (JPEG/PNG) — max 10 MB", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# Optional: let user paste an image URL
with st.expander("Or use an image URL (paste link)"):
    image_url = st.text_input("Image URL", value="")
    if image_url:
        st.caption("The app will fetch the image from the provided URL when you press 'Load from URL'.")
        if st.button("Load from URL"):
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                uploaded_file = io.BytesIO(resp.content)
                uploaded_file.name = "from_url.jpg"
                st.success("Image loaded from URL."
                           )
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")


# Show preview and image info if file present
if uploaded_file is not None:
    valid, err = is_valid_image_file(uploaded_file)
    if not valid:
        st.error(err)
        st.stop()

    # Safety: reset pointer and open image
    uploaded_file.seek(0)
    image = Image.open(io.BytesIO(uploaded_file.read()))
    uploaded_file.seek(0)

    cols = st.columns([1, 1])
    with cols[0]:
        st.subheader("Preview")
        st.image(image, width='stretch')
    with cols[1]:
        st.subheader("Image details")
        st.write(f"Filename: {getattr(uploaded_file, 'name', 'uploaded_image')} ")
        try:
            st.write(f"Dimensions: {image.width} x {image.height} px")
        except Exception:
            pass
        try:
            file_size_kb = bytes_to_kb(uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue()))
            st.write(f"Size: {file_size_kb:.0f} KB")
        except Exception:
            pass

    st.markdown("---")

    # Analysis options
    analysis_mode = st.radio("Select analysis mode:", ("Quick Prediction (computer vision)", "Full Analysis (prediction + AI explanation)"))

    # Buttons
    run_col, clear_col = st.columns([1, 1])
    with run_col:
        run_analysis = st.button("Start analysis")
    with clear_col:
        if st.button("Clear selection"):
            # Clear page by rerunning without file
            st.rerun()

    if run_analysis:
        # Check API availability first
        healthy, info = test_api_health(api_base_url)
        if not healthy:
            st.error(f"Backend is unreachable: {info}")
        else:
            # Choose endpoint
            if analysis_mode.startswith("Quick"):
                endpoint = f"{api_base_url.rstrip('/')}/predict-image"
                timeout = 30
            else:
                endpoint = f"{api_base_url.rstrip('/')}/predict-and-explain"
                timeout = 90

            st.info("Sending image to backend. This operation may take several seconds depending on model size and network.")
            with st.spinner("Analyzing — please wait..."):
                # Reset file pointer
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass

                data, error = post_file(endpoint, uploaded_file, timeout=timeout)

            if error:
                st.error(f"Analysis failed — {error}")
            else:
                # Handle responses for both endpoints consistently
                try:
                    if analysis_mode.startswith("Quick"):
                        # Expected: { "label": "...", "confidence": 0.92 }
                        label = data.get("label")
                        confidence = float(data.get("confidence", 0.0))
                        st.success("Prediction complete")
                        st.subheader("Detection Result")
                        st.write(f"**Condition:** {label}")
                        st.write(f"**Confidence:** {confidence * 100:.1f}%")

                        if st.button("Get AI explanation for this result"):
                            # Call the explain endpoint
                            st.info("Requesting explanation from the AI module...")
                            expl_endpoint = f"{api_base_url.rstrip('/')}/explain-result"
                            try:
                                expl_payload = {"label": label, "confidence": confidence}
                                resp = requests.post(expl_endpoint, json=expl_payload, timeout=60)
                                if resp.status_code == 200:
                                    expl = resp.json()
                                    st.subheader("AI Explanation & Recommendations")
                                    st.markdown(expl.get("explanation", ""), unsafe_allow_html=False)
                                else:
                                    st.error(f"Explanation failed: {resp.status_code} — {resp.text}")
                            except Exception as e:
                                st.error(f"Explanation request error: {e}")

                    else:
                        # Full analysis: expected structure may include classification + explanation
                        classification = data.get("classification") or {}
                        explanation = data.get("explanation") or data.get("analysis") or ""
                        label = classification.get("label") or data.get("label")
                        confidence = float(classification.get("confidence", data.get("confidence", 0.0)))
                        assessment = data.get("confidence_level") or classification.get("assessment") or ""

                        st.success("Full analysis complete")
                        st.subheader("Detection Summary")
                        st.write(f"**Condition:** {label}")
                        st.write(f"**Confidence:** {confidence * 100:.1f}%")
                        if assessment:
                            st.write(f"**Assessment:** {assessment}")

                        st.markdown("---")
                        st.subheader("AI Analysis & Recommendations")
                        st.markdown(explanation, unsafe_allow_html=False)

                        # Download report
                        report = (
                            f"PALM OIL DISEASE REPORT\n"
                            f"Condition: {label}\n"
                            f"Confidence: {confidence * 100:.1f}%\n\n"
                            f"Recommendations:\n{explanation}\n"
                        )
                        st.download_button("Download text report", data=report, file_name=f"palm_report_{(label or 'result').replace(' ', '_')}.txt")

                except Exception as e:
                    logger.exception("Failed to render result")
                    st.error(f"Unexpected response format from backend: {e}")

else:
    st.info("No image selected. Upload a photo or paste an image URL to get started.")


# ----------------------------
# Footer / Troubleshooting
# ----------------------------

st.markdown("---")
with st.expander("Troubleshooting & Tips"):
    st.write(
        """
- Use a clear, well-lit photo focused on the affected leaf area.
- If the API is unreachable, ensure your backend is running and the API Base URL is correct.
- For large images, reduce resolution before uploading to stay under 10 MB.
- If responses are slow, try Quick Prediction first to validate the image, then request Full Analysis.
"""
    )


# End of file
