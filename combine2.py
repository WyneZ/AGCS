# Python In-built packages
from pathlib import Path
import PIL
import cv2
# import pafy  # disable if not used
# import pickle

# External packages
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Setting page layout
st.set_page_config(
    page_title="Waste Classification using YOLOv8",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .main-header { font-size: 3rem; color: #004D73; text-align: center; margin-bottom: 1rem; font-weight: 600; }
    .sub-header { font-size: 1.5rem; color: #37A67A; border-bottom: 2px solid #FFC107; padding-bottom: 0.5rem; margin-top: 1.5rem; }
    .info-box { background-color: #E0F2F7; border-radius: 10px; padding: 15px; margin: 10px 0;
                border-left: 5px solid #37A67A; font-size: 1.1rem; color: #004D73; }
    .stButton button { background-color: #004D73; color: white; border-radius: 5px; padding: 0.5rem 1rem;
                       border: none; width: 100%; font-size: 1.1rem; }
    .stButton button:hover { background-color: #FFC107; color: #004D73; }
    .metric-card { background-color: #F0F4F7; padding: 15px; border-radius: 10px; text-align: center;
                   box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: 10px 0; border-left: 5px solid #004D73; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #37A67A; }
    .metric-label { font-size: 1.1rem; color: #5C5C5C; }
</style>
""", unsafe_allow_html=True)

# Default paths
DETECTION_MODEL = 'weights/yoloooo.pt'
DEFAULT_IMAGE = 'images/def.jfif'
SOURCES_LIST = ["Image", "Webcam (Snapshot)", "Webcam (Live)", "Video"]

# Helper functions
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=False, tracker=None):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    results = model.predict(image, conf=conf)
    res_plotted = results[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)

# --- Webcam Snapshot Mode ---
def play_webcam_snapshot(conf, model):
    st.info("📸 Capture an image from your webcam")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(cv2_img, conf=conf)
        res_plotted = results[0].plot()[:, :, ::-1]

        st.image(res_plotted, caption="Detected Objects", use_container_width=True)

# --- Webcam Live Mode (streamlit-webrtc) ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.conf)
        return results[0].plot()

def play_webcam_live(conf, model):
    st.info("🎥 Live webcam detection (requires browser permission)")
    webrtc_streamer(
        key="example",
        video_transformer_factory=lambda: VideoTransformer(model, conf)
    )

# --- Video File Detection ---
def handle_video_detection(conf, model):
    video_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
    if video_file:
        st.video(video_file)
        if st.sidebar.button('Detect Objects in Video'):
            temp_path = Path("temp_uploaded_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            vid_cap = cv2.VideoCapture(str(temp_path))
            st_frame = st.empty()
            progress_bar = st.progress(0)
            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            st.info("Processing video...")
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    current_frame += 1
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    st.success("Video processing finished.")
                    vid_cap.release()
                    progress_bar.empty()
                    break
            if temp_path.exists():
                temp_path.unlink()

# --- Main ---
st.markdown('<h1 class="main-header">♻️ Waste Classification using YOLOv8</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This app uses <b>YOLOv8</b> to detect and classify waste.</p>
    <p>Choose Image, Webcam, or Video from the sidebar to start.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)
confidence = float(st.sidebar.slider("Confidence", 25, 100, 40)) / 100

model = load_model(Path(DETECTION_MODEL))
if model is None:
    st.stop()

st.sidebar.markdown('<div class="sub-header">Source</div>', unsafe_allow_html=True)
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)

if source_radio == "Image":
    st.markdown('<div class="sub-header">Image Detection</div>', unsafe_allow_html=True)
    source_img = st.file_uploader("Upload an image...", type=("jpg","jpeg","png","bmp","webp"))
    if source_img is not None:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        if st.button('Detect Objects', key="detect_objects"):
            results = model.predict(uploaded_image, conf=confidence)
            res_plotted = results[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Objects', use_container_width=True)
    else:
        try:
            default_image = PIL.Image.open(DEFAULT_IMAGE)
            st.image(default_image, caption="Example Image", use_container_width=True)
        except:
            st.error("Error loading default image.")

elif source_radio == "Webcam (Snapshot)":
    st.markdown('<div class="sub-header">Webcam Snapshot</div>', unsafe_allow_html=True)
    play_webcam_snapshot(confidence, model)

elif source_radio == "Webcam (Live)":
    st.markdown('<div class="sub-header">Webcam Live Stream</div>', unsafe_allow_html=True)
    play_webcam_live(confidence, model)

elif source_radio == "Video":
    st.markdown('<div class="sub-header">Video Detection</div>', unsafe_allow_html=True)
    handle_video_detection(confidence, model)

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center; color:#5C5C5C;">Waste Classification App using YOLOv8 | Made with Streamlit</div>', unsafe_allow_html=True)
