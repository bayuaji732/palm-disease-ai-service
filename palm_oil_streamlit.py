# palm_oil_streamlit.py
import streamlit as st
import requests
from PIL import Image
import io
import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

# PDF generation imports
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Image as RLImage, 
    ListFlowable, 
    ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

import markdown
import re

# ----------------------------
# Configuration / Constants
# ----------------------------
DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
REQUEST_TIMEOUT = 60  # seconds
APP_TITLE = "Palm Oil Disease Detection"
TIMEZONE = "Asia/Jakarta"  # for timestamp in report

# Streamlit page config
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Lightweight logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("palm_app")


# ----------------------------
# Helper functions
# ----------------------------

def bytes_to_kb(n: int) -> float:
    return n / 1024.0 if n is not None else 0.0


def is_valid_image_file(file) -> Tuple[bool, Optional[str]]:
    """Basic validation for uploaded file size and type."""
    if file is None:
        return False, "No file provided."

    # If object has 'size' attribute (Streamlit's UploadedFile) use that, else attempt to get length of getvalue()
    try:
        size = getattr(file, "size", None)
        if size is None:
            # file may be BytesIO
            size = len(file.getvalue())
    except Exception:
        size = None

    if size is not None and size > MAX_FILE_SIZE:
        return False, f"File too large: {bytes_to_kb(size):.0f} KB (max {bytes_to_kb(MAX_FILE_SIZE):.0f} KB)"

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
        # Ensure pointer is at start
        try:
            file.seek(0)
        except Exception:
            pass

        # Determine filename and content
        filename = getattr(file, "name", "image.jpg")
        file_bytes = file.read() if hasattr(file, "read") else file.getvalue()
        files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
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
    
def markdown_to_reportlab_blocks(md_text: str, styles):
    """
    Parses Markdown and converts to ReportLab flowables with TIGHT spacing.
    """
    flowables = []
    if not md_text:
        return flowables

    # Convert Markdown bold **text** to HTML <b>text</b>
    # and *italics* to <i>italics</i>
    md_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md_text)
    md_text = re.sub(r'(?<!\s)\*(.*?)\*(?!\s)', r'<i>\1</i>', md_text)

    lines = md_text.splitlines()
    text_buffer = []

    # Helper to flush standard text
    def flush_buffer():
        if text_buffer:
            combined_text = " ".join(text_buffer)
            flowables.append(Paragraph(combined_text, styles["Normal"]))
            # Reduced spacing after paragraphs (was 8, now 3)
            flowables.append(Spacer(1, 3))
            text_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            flush_buffer()
            # Reduced empty line spacing (was 8, now 4)
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # --- Headers (## or ###) ---
        if line.startswith("#"):
            flush_buffer()
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            
            # Select style based on level
            if level == 1: s = styles["Heading1"]
            elif level == 2: s = styles["Heading2"]
            else: s = styles["Heading3"]
            
            # Reduced spacing around headers
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph(text, s))
            flowables.append(Spacer(1, 2))
            i += 1
            continue

        # --- Lists (Bullets or Numbers) ---
        # Detect start of a list
        is_bullet = line.startswith(("* ", "- "))
        is_number = re.match(r"^\d+\.\s+", line)

        if is_bullet or is_number:
            flush_buffer()
            list_items = []
            list_start_value = None  # Variable to capture the start number of the first item
            
            # Look ahead to capture all sequential list items
            while i < len(lines):
                sub_line = lines[i].strip()
                sub_is_bullet = sub_line.startswith(("* ", "- "))
                sub_match = re.match(r"^(\d+)\.\s+(.*)", sub_line)
                
                if not (sub_is_bullet or sub_match):
                    break # End of list
                
                # Format content
                if sub_is_bullet:
                    content = sub_line[2:].strip()
                    current_val = None
                else:
                    content = sub_match.group(2).strip()
                    current_val = int(sub_match.group(1))

                # Capture start value from the very first item
                if list_start_value is None and current_val is not None:
                    list_start_value = current_val

                list_items.append(ListItem(
                    Paragraph(content, styles["Normal"]),
                    value=current_val
                ))
                i += 1
            
            # Create the compact list block
            t_list = ListFlowable(
                list_items,
                bulletType='bullet' if is_bullet else '1',
                start=list_start_value if list_start_value is not None else 1,
                leftIndent=12,
                bulletFontSize=10,
                # These control vertical spacing inside the list
                spaceBefore=2,
                spaceAfter=2
            )
            flowables.append(t_list)
            flowables.append(Spacer(1, 4))
            continue

        # --- Standard Text ---
        text_buffer.append(line)
        i += 1

    flush_buffer()
    return flowables

def generate_pdf_report(
    label: str,
    confidence: float,
    assessment: str,
    explanation: str,
    timestamp: datetime,
    image_bytes: Optional[bytes] = None,
    api_base_url: Optional[str] = None,
    api_version: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF report and return it as bytes.

    - label: detected condition
    - confidence: between 0 and 1
    - assessment: optional assessment text
    - explanation: potentially multi-paragraph recommendations
    - timestamp: datetime for the report
    - image_bytes: optional bytes of the uploaded image to include as a thumbnail
    - api_base_url / api_version: optional diagnostic info
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    small_center = ParagraphStyle(
        "SmallCenter",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
    )

    story.append(Paragraph("Palm Oil Disease Detection Report", title_style))
    # small subtitle with timestamp
    tzname = TIMEZONE
    ts_str = timestamp.astimezone(ZoneInfo(tzname)).strftime("%Y-%m-%d %H:%M:%S %Z")
    story.append(Paragraph(f"Analysis timestamp: {ts_str}", small_center))
    story.append(Spacer(1, 12))

    # Insert image thumbnail if available (scaled to fit page width)
    if image_bytes:
        try:
            img_buffer = io.BytesIO(image_bytes)
            pil_img = Image.open(img_buffer)

            img_width, img_height = pil_img.size
            max_width = A4[0] - (4 * cm)

            scale = min(1.0, max_width / img_width)
            display_width = img_width * scale
            display_height = img_height * scale

            img_buffer.seek(0)  # IMPORTANT: rewind buffer

            rl_img = RLImage(
                img_buffer,
                width=display_width,
                height=display_height,
                kind='proportional' # Add this for better aspect ratio handling
            )

            story.append(rl_img)
            story.append(Spacer(1, 12))

        except Exception:
            logger.exception("Failed to render thumbnail into PDF")

    # Summary block
    summary_html = f"""
    <b>Detected Condition:</b> {label or 'Unknown'}<br/>
    <b>Confidence:</b> {confidence * 100:.1f}%<br/>
    """
    if assessment:
        summary_html += f"<b>Assessment:</b> {assessment}"
    
    story.append(Paragraph(summary_html, styles["Normal"]))
    story.append(Spacer(1, 10)) # Space before the main AI analysis

    story.append(Paragraph("<b>AI Analysis & Recommendations:</b>", styles["Heading2"]))
    story.append(Spacer(1, 4)) # Reduced from 8

    # Process AI explanation with the new tight parser
    blocks = markdown_to_reportlab_blocks(explanation, styles)
    story.extend(blocks)

    # Footer: diagnostic info (optional)
    story.append(Spacer(1, 18))
    footer_lines = []
    if api_base_url:
        footer_lines.append(f"API Base URL: {api_base_url}")
    if api_version:
        footer_lines.append(f"API version: {api_version}")
    footer_lines.append(f"Generated by: {APP_TITLE}")
    footer_lines.append(f"Generated on: {ts_str}")

    for line in footer_lines:
        story.append(Paragraph(line, ParagraphStyle("footer", parent=styles["Normal"], fontSize=8)))
        story.append(Spacer(1, 2))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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

st.title(APP_TITLE)
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
                st.success("Image loaded from URL.")
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")


# Show preview and image info if file present
if uploaded_file is not None:
    valid, err = is_valid_image_file(uploaded_file)
    if not valid:
        st.error(err)
        st.stop()

    # Safety: reset pointer and open image
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        # Ensure file pointer reset for later network post
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    except Exception:
        st.error("Unable to open uploaded image.")
        st.stop()

    cols = st.columns([1, 1])
    with cols[0]:
        st.subheader("Preview")
        # Use use_column_width for proper responsive display
        st.image(image, width="stretch")
    with cols[1]:
        st.subheader("Image details")
        st.write(f"Filename: {getattr(uploaded_file, 'name', 'uploaded_image')}")
        try:
            st.write(f"Dimensions: {image.width} x {image.height} px")
        except Exception:
            pass
        try:
            file_size_kb = bytes_to_kb(getattr(uploaded_file, "size", None) or len(uploaded_file.getvalue()))
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
            st.experimental_rerun()

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
                # Reset file pointer for reading and posting
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
                        label = classification.get("label") or data.get("label") or "Unknown"
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

                        # Prepare PDF bytes
                        # Make sure to read image bytes again (file pointer may be at end)
                        try:
                            uploaded_file.seek(0)
                        except Exception:
                            pass
                        try:
                            image_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getvalue()
                        except Exception:
                            image_bytes = None

                        timestamp = datetime.now(tz=ZoneInfo(TIMEZONE))
                        api_ver = info if info else None

                        pdf_bytes = generate_pdf_report(
                            label=label,
                            confidence=confidence,
                            assessment=assessment,
                            explanation=explanation,
                            timestamp=timestamp,
                            image_bytes=image_bytes,
                            api_base_url=api_base_url,
                            api_version=api_ver,
                        )

                        st.download_button(
                            label="Download PDF report",
                            data=pdf_bytes,
                            file_name=f"palm_report_{(label or 'result').replace(' ', '_')}.pdf",
                            mime="application/pdf",
                        )

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
