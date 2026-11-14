import cv2
import numpy as np
import streamlit as st

from io import BytesIO
from PIL import Image

from face_engine import FaceDetector
from recognition_engine import FaceRecognitionManager


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FaceVision | Intelligent Face System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING AND HTML HELPERS
# ============================================================

def html(block):
    """Render a compact HTML block with Streamlit."""
    st.markdown(block.strip(), unsafe_allow_html=True)


html("""
<style>
.stApp {
    background-color: #eef2f7;
}

.block-container {
    max-width: 1400px;
    padding-top: 4rem;
    padding-bottom: 2rem;
}

.block-container p,
.block-container li,
.block-container span {
    color: #334155;
}

.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4 {
    color: #0f172a;
}

[data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1b2d;
}

[data-testid="stSidebar"] * {
    color: #e8eef7;
}

[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color: #2b3d57;
}

.sidebar-brand {
    padding: 10px 4px 14px;
}

.sidebar-brand h1 {
    color: #ffffff !important;
    font-size: 31px;
    font-weight: 800;
    margin: 0;
}

.sidebar-brand p {
    color: #9fb3cc !important;
    font-size: 14px;
    line-height: 1.5;
    margin: 8px 0 0;
}

.sidebar-guide {
    background: #17263c;
    border: 1px solid #2b3d57;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    margin-top: 10px;
    padding: 18px;
}

.sidebar-guide h3 {
    color: #ffffff !important;
    font-size: 19px;
    margin: 0 0 14px;
}

.sidebar-guide p {
    color: #cbd9ea !important;
    font-size: 14px;
    line-height: 1.6;
    margin: 0 0 12px;
}

.sidebar-guide b {
    color: #38bdf8 !important;
}

.sidebar-footer {
    color: #7f93ad !important;
    font-size: 12px;
    line-height: 1.6;
    margin-top: 30px;
    text-align: center;
}

/* Hero and headings */
.hero {
    background: linear-gradient(115deg, #0f2a5c 0%, #1d4ed8 100%);
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(29, 78, 216, 0.25);
    margin-bottom: 28px;
    padding: 34px 38px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 44px;
    font-weight: 800;
    margin: 0 0 10px;
}

.hero p {
    color: #e0ecff !important;
    font-size: 18px;
    line-height: 1.6;
    margin: 0;
    max-width: 850px;
}

.section-heading {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    margin: 10px 0 18px;
    padding-left: 0;
}

/* Cards and panels */
.feature-card {
    background: #ffffff;
    border: 2px solid #1d4ed8;
    border-radius: 14px;
    box-sizing: border-box;
    height: 190px;
    padding: 22px 24px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    
}

.feature-card h3 {
    color: #0f172a !important;
    font-size: 23px !important;
    margin: 0 0 12px;
}

.feature-card p {
    color: #475569 !important;
    font-size: 16px;
    line-height: 1.6;
    margin: 0;
}

.info-panel,
.page-header {
    background: #ffffff;
    border: 1px solid #d5deea;
    border-radius: 14px;
    padding: 22px 26px;
}

.info-panel {
    border-left: 5px solid #38bdf8;
    margin-top: 20px;
}

.info-panel h3 {
    color: #0f172a !important;
    font-size: 23px !important;
    margin: 0 0 10px;
}

.info-panel p {
    color: #475569 !important;
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
}

.page-header {
    border-left: 6px solid #1d4ed8;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    margin-bottom: 26px;
}

.page-header h2 {
    color: #0f172a !important;
    font-size: 32px !important;
    font-weight: 800;
    margin: 0 0 8px;
}

.page-header p {
    color: #475569 !important;
    font-size: 16px;
    line-height: 1.6;
    margin: 0;
}

/* Inputs and buttons */
.stTextInput input {
    background: #ffffff;
    border: 1.5px solid #94a3b8;
    border-radius: 10px;
    color: #0f172a;
    font-size: 16px;
    min-height: 46px;
}

.stTextInput input::placeholder {
    color: #64748b;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-size: 16px;
    font-weight: 700;
    min-height: 48px;
}

.stButton > button[kind="primary"] {
    background: #1d4ed8;
    border: none;
    color: #ffffff;
}

.stButton > button[kind="primary"]:hover {
    background: #1e40af;
    color: #ffffff;
}

.stButton > button[kind="primary"] p {
    color: #ffffff !important;
}

.stDownloadButton > button {
    background: #ffffff;
    border: 2px solid #1d4ed8;
    color: #1d4ed8;
}

.stDownloadButton > button p {
    color: #1d4ed8 !important;
}

/* Metrics and uploaders */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d5deea;
    border-bottom: 4px solid #1d4ed8;
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-size: 16px !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 34px !important;
    font-weight: 800;
}

[data-testid="stFileUploader"] section {
    background: #ffffff;
    border: 2px dashed #94a3b8;
    border-radius: 12px;
}

[data-testid="stFileUploader"] section * {
    color: #334155 !important;
}

[data-testid="stFileUploader"] button {
    background: #1d4ed8;
    border: none;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}
.about-facevision-box {
    border: 2px solid #1d4ed8;
    border-radius: 14px;
}
.privacy-note {
  margin-top: 12px;
}
.footer {
    color: #64748b;
    font-size: 14px;
    line-height: 1.7;
    padding: 28px 0 6px;
    text-align: center;
     margin-top: 12px;

</style>
""")


# ============================================================
# ENGINES
# ============================================================

@st.cache_resource
def get_detector():
    return FaceDetector()


@st.cache_resource
def get_manager():
    return FaceRecognitionManager()


try:
    detector = get_detector()
    manager = get_manager()
except Exception as error:
    st.error(f"Could not initialize FaceVision: {error}")
    st.stop()


# ============================================================
# SHARED HELPERS
# ============================================================

def uploaded_image_to_bgr(uploaded_file):
    """Convert an uploaded image to a BGR NumPy array."""
    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb_image = np.array(pil_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def display_bgr_image(image, caption=None):
    """Display a BGR image in Streamlit."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption=caption, use_column_width=True)

def get_registration_statistics():
    """Return the number of people and saved face samples."""
    people = 0
    samples = 0

    if not manager.faces_dir.exists():
        return people, samples

    for folder in manager.faces_dir.iterdir():
        if not folder.is_dir():
            continue

        images = list(folder.glob("*.jpg"))
        if images:
            people += 1
            samples += len(images)

    return people, samples


def get_model_status():
    """Return whether both model files exist."""
    return manager.model_path.exists() and manager.labels_path.exists()


def page_header(title, description):
    html(f"""
    <div class="page-header">
        <h2>{title}</h2>
        <p>{description}</p>
    </div>
    """)


def feature_card(title, description):
    html(f"""
    <div class="feature-card">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """)


def render_download_button(image, label, filename):
    """Encode and offer a processed image as a JPEG download."""
    success, encoded_image = cv2.imencode(".jpg", image)

    if success:
        st.download_button(
            label,
            data=encoded_image.tobytes(),
            file_name=filename,
            mime="image/jpeg",
            use_container_width=True,
        )


def render_hero():
    html("""
    <div class="hero">
        <h1>👁️ FaceVision</h1>
        <p>
            Intelligent face detection and recognition system.
            Register faces, train your model and analyze images
            through one interactive dashboard.
        </p>
    </div>
    """)


# ============================================================
# PAGE CONTENT
# ============================================================

def render_dashboard():
    html('<div class="section-heading">System Overview</div>')

    people_count, sample_count = get_registration_statistics()
    model_ready = get_model_status()

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.metric("Registered People", people_count)

    with col2:
        st.metric("Face Samples", sample_count)

    with col3:
        st.metric(
            "Recognition Model",
            "Ready" if model_ready else "Not Trained",
        )

    st.write("")
    html('<div class="section-heading">Available Features</div>')

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        feature_card(
            "📸 Face Registration",
            "Register people by uploading clear face photographs.",
        )

    with col2:
        feature_card(
            "🧠 Face Recognition",
            "Identify registered faces using the trained LBPH model.",
        )

    st.write("")
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        feature_card(
            "🔍 Face Detection",
            "Detect and locate faces in uploaded photographs.",
        )

    with col4:
        feature_card(
            "⚙️ Model Training",
            "Train the recognition model using registered samples.",
        )

    st.write("")

    if model_ready:
        st.success("Your recognition model is trained and ready to use.")
    else:
        st.info(
            "Getting started: Register face samples, then train your "
            "recognition model before using Face Recognition."
        )

    html("""
    <div class="info-panel">
        <h3>About FaceVision</h3>
        <p>
            FaceVision combines Haar Cascade face detection with
            Local Binary Patterns Histograms (LBPH) recognition
            to detect and identify faces in images.
        </p>
    </div>
    """)

    st.caption(
        "Privacy note: face samples are stored in the local data directory. "
        "Only register images you have permission to use."
    )

def render_registration():
    page_header(
        "📸 Face Registration",
        "Add a person by uploading a photograph containing exactly one face.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        person_name = st.text_input(
            "Person's Name",
            placeholder="Enter the person's name",
            key="registration_person_name",
        )

        registration_file = st.file_uploader(
            "Upload Face Photograph",
            type=["jpg", "jpeg", "png"],
            key="registration_upload",
        )

        register_clicked = st.button(
            "📸 Register Face",
            type="primary",
            use_container_width=True,
            disabled=(
                not person_name.strip()
                or registration_file is None
            ),
        )

    with right:
        if registration_file is None:
            st.info("Upload a photograph to see its preview here.")
        else:
            try:
                image = uploaded_image_to_bgr(registration_file)

                st.markdown("#### Photograph Preview")
                display_bgr_image(image, "Uploaded photograph")

            except Exception as error:
                st.error(f"Could not read the uploaded image: {error}")
                return

    if register_clicked:
        try:
            image = uploaded_image_to_bgr(registration_file)

            with st.spinner("Registering face..."):
                saved_path = manager.register_face(
                    image,
                    person_name.strip(),
                )

            st.success("Face registered successfully!")
            st.caption(f"Saved sample: {saved_path}")

        except Exception as error:
            st.error(f"Registration failed: {error}")

    st.caption(
        "Face samples are saved locally in the application's data directory. "
        "Register only images you have permission to use."
    )


def render_training():
    page_header(
        "⚙️ Model Training",
        "Train the LBPH recognition model using registered face samples.",
    )

    people_count, sample_count = get_registration_statistics()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.metric("Registered People", people_count)

    with col2:
        st.metric("Available Samples", sample_count)

    if sample_count == 0:
        st.info("Register at least one face sample before training.")

    if st.button(
        "⚙️ Train Model",
        type="primary",
        use_container_width=True,
        disabled=(sample_count == 0),
    ):
        try:
            with st.spinner("Training recognition model..."):
                total_samples, total_people = manager.train_model()

            st.success("Recognition model trained successfully!")

            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.metric("People Trained", total_people)

            with col2:
                st.metric("Training Samples", total_samples)

        except Exception as error:
            st.error(str(error))


def render_detection():
    page_header(
        "🔍 Face Detection",
        "Upload an image and let FaceVision detect and locate faces.",
    )

    detection_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
    )

    if detection_file is None:
        return

    try:
        image = uploaded_image_to_bgr(detection_file)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("#### Original Photograph")
            display_bgr_image(image)

        with right:
            st.markdown("#### Detection Controls")
            st.write(
                "Scan the photograph and highlight each detected face."
            )
            run_detection = st.button(
                "🔍 Detect Faces",
                type="primary",
                use_container_width=True,
            )

        if not run_detection:
            return

        with st.spinner("Detecting faces..."):
            result_image, faces = detector.detect_faces(image)

        st.divider()
        st.markdown("### Detection Results")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.metric("Faces Detected", len(faces))

        with col2:
            st.metric("Detection Status", "Completed")

        if len(faces) > 0:
            st.success(f"Successfully detected {len(faces)} face(s).")
        else:
            st.warning(
            "No faces were detected. Try a clearer photograph "
            "with better lighting."
    )

        st.markdown("#### Processed Photograph")
        display_bgr_image(
            result_image,
            "Detected faces highlighted with bounding boxes",
        )

        render_download_button(
            result_image,
            "⬇️ Download Detection Result",
            "face_detection_result.jpg",
        )

    except Exception as error:
        st.error(f"Detection error: {error}")


def render_recognition():
    page_header(
        "🧠 Face Recognition",
        "Identify registered people by comparing faces with your trained model.",
    )

    model_ready = get_model_status()

    if not model_ready:
        st.warning(
            "Your recognition model is not trained yet. "
            "Register face samples and train the model first."
        )

    recognition_file = st.file_uploader(
        "Upload Recognition Image",
        type=["jpg", "jpeg", "png"],
        key="recognition_upload",
    )

    threshold = st.slider(
        "Recognition Distance Threshold",
        min_value=30,
        max_value=120,
        value=70,
        step=5,
        help=(
            "Lower LBPH distance means a closer match. "
            "This threshold is experimental, not a confidence percentage."
        ),
    )

    if recognition_file is None:
        return

    try:
        image = uploaded_image_to_bgr(recognition_file)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("#### Uploaded Photograph")
            display_bgr_image(image)

        with right:
            st.markdown("#### Recognition Controls")
            st.write(
                "Detect faces and compare them with registered people."
            )
            run_recognition = st.button(
                "🧠 Recognize Faces",
                type="primary",
                use_container_width=True,
                disabled=not model_ready,
            )

        if not run_recognition:
            return

        with st.spinner("Analyzing faces..."):
            result_image, results = manager.recognize_faces(
                image,
                threshold=threshold,
            )

        st.divider()
        st.markdown("### Recognition Results")

        total_faces = len(results)
        recognized = sum(
            result["name"] != "Unknown" for result in results
        )
        unknown = total_faces - recognized

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.metric("Faces Detected", total_faces)

        with col2:
            st.metric("Recognized", recognized)

        with col3:
            st.metric("Unknown", unknown)

        if total_faces == 0:
            st.warning("No faces were detected in this photograph.")
        elif recognized:
            st.success(f"Successfully recognized {recognized} face(s).")
        else:
            st.warning(
                "Faces were detected, but no registered person matched "
                "the selected threshold."
            )

        st.markdown("#### Individual Results")

        for index, result in enumerate(results, start=1):
            with st.container(border=True):
                col1, col2 = st.columns([2, 1], gap="medium")

                with col1:
                    if result["name"] == "Unknown":
                        st.warning(f"Face {index}: Unknown Person")
                    else:
                        st.success(f"Face {index}: {result['name']}")

                with col2:
                    st.markdown(
                        f"**LBPH Distance:** {result['distance']}"
                    )

        st.markdown("#### Recognition Output")
        display_bgr_image(
            result_image,
            "Recognition results with labeled face regions",
        )

        render_download_button(
            result_image,
            "⬇️ Download Recognition Result",
            "face_recognition_result.jpg",
        )

    except Exception as error:
        st.error(f"Recognition error: {error}")


# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "Dashboard",
    "Face Registration",
    "Model Training",
    "Image Detection",
    "Face Recognition",
]

with st.sidebar:
    html("""
    <div class="sidebar-brand">
        <h1>👁️ FaceVision</h1>
        <p>Intelligent Face Detection &amp; Recognition</p>
    </div>
    """)

    st.divider()
    st.markdown("### Navigation")

    page = st.radio(
        "Choose a section",
        PAGES,
        label_visibility="collapsed",
    )

    st.divider()

    html("""
    <div class="sidebar-guide">
        <h3> Getting Started</h3>
        <p><b>01. Register Faces</b><br>Add clear photographs of people.</p>
        <p><b>02. Train the Model</b><br>Train using registered face samples.</p>
        <p><b>03. Detect &amp; Recognize</b><br>Analyze photographs and identify registered people.</p>
        <p style="margin-bottom:0;">
            💡 Use multiple clear photographs per person for better testing.
        </p>
    </div>
    """)

    html("""
    <div class="sidebar-footer">
        FACEVISION PROJECT<br>
        Built with Python &amp; OpenCV
    </div>
    """)


# ============================================================
# APPLICATION
# ============================================================

render_hero()

if page == "Dashboard":
    render_dashboard()
elif page == "Face Registration":
    render_registration()
elif page == "Model Training":
    render_training()
elif page == "Image Detection":
    render_detection()
elif page == "Face Recognition":
    render_recognition()

html("""
<div class="footer">
    FaceVision | Intelligent Face Detection &amp; Recognition<br>
    Developed using Python, OpenCV and Streamlit
</div>
""")