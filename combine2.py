# Python In-built packages
from pathlib import Path
import PIL
import cv2
import pafy
# import pickle

# External packages
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="Waste Classification using YOLOv8",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        color: #004D73;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #37A67A;
        border-bottom: 2px solid #FFC107;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #E0F2F7; /* Changed to a light blue */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #37A67A;
        font-size: 1.1rem;
        color: #004D73; /* Ensuring text is readable against light blue */
    }
    .stButton button {
        background-color: #004D73;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background-color: #FFC107;
        color: #004D73;
    }
    .metric-card {
        background-color: #F0F4F7;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 5px solid #004D73;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #37A67A;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #5C5C5C;
    }
</style>
""", unsafe_allow_html=True)

# Default paths
DETECTION_MODEL = 'weights/yoloooo.pt'
SEGMENTATION_MODEL = ''
DEFAULT_IMAGE = 'images/def.jfif'
WEBCAM_PATH = 0
SOURCES_LIST = ["Image", "Webcam", "Video"]

# Helper functions
def load_model(model_path):
    """Loads a YOLO object detection model from the specified model_path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model from {model_path}: {e}")
        return None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=False, tracker=None):
    """
    Processes and displays video frames with object detection or tracking.
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    if is_display_tracking:
        results = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        results = model.predict(image, conf=conf)

    res_plotted = results[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)

def play_youtube_video(conf, model):
    """Handles real-time object detection on a YouTube video stream."""
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    display_tracker = st.sidebar.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = (display_tracker == 'Yes')
    tracker = None
    if is_display_tracker:
        tracker = st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))

    if st.sidebar.button('Detect Objects in YouTube Video'):
        if not source_youtube:
            st.sidebar.warning("Please enter a YouTube video URL.")
            return

        try:
            video = pafy.new(source_youtube)
            best = video.getbest(preftype="mp4")
            vid_cap = cv2.VideoCapture(best.url)
            st_frame = st.empty()
            
            st.info("Loading and processing YouTube video...")
            
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    st.success("YouTube video processing finished.")
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error loading video: {str(e)}")

def play_webcam(conf, model):
    """Handles real-time object detection on a webcam stream."""
    is_display_tracker = False
    tracker = None
    
    st.info("Webcam feed will start when you click the button below. Make sure your webcam is connected and accessible.")
    
    if st.sidebar.button('Start Webcam Detection'):
        try:
            vid_cap = cv2.VideoCapture(WEBCAM_PATH)
            if not vid_cap.isOpened():
                st.error("Error: Could not open webcam. Please check permissions or if another application is using it.")
                return
                
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    st.warning("Webcam stream ended.")
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error accessing webcam: {str(e)}")

def handle_video_detection(conf, model):
    """Handles object detection on an uploaded video file."""
    is_display_tracker = False
    tracker = None
    
    video_file = st.sidebar.file_uploader("Upload a video file...", type=["mp4", "mov", "avi"])

    if video_file:
        st.video(video_file)
        
        if st.sidebar.button('Detect Objects in Video'):
            try:
                temp_path = Path("temp_uploaded_video.mp4")
                with open(temp_path, "wb") as f:
                    f.write(video_file.read())
                
                vid_cap = cv2.VideoCapture(str(temp_path))
                st_frame = st.empty()
                progress_bar = st.progress(0)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                st.info("Processing video frames...")

                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        current_frame += 1
                        progress_bar.progress(min(current_frame / total_frames, 1.0))
                        _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                    else:
                        st.success("Video processing finished.")
                        vid_cap.release()
                        progress_bar.empty()
                        break
                
                if temp_path.exists():
                    temp_path.unlink()  # Clean up the temporary file
            except Exception as e:
                st.sidebar.error(f"Error loading video: {str(e)}")

# --- Main Application Logic ---
st.markdown('<h1 class="main-header">♻️ Waste Classification using YOLOv8</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This application leverages the power of <b>YOLOv8</b> to accurately detect and classify various types of waste materials.</p>
    <p>To begin, simply select a source (Image, Webcam, or Video) from the sidebar and upload your content.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
# st.sidebar.markdown('<h2 style="color: #004D73;">Configuration</h2>', unsafe_allow_html=True)

# Model Options
st.sidebar.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Load models
#try:
#    with open('weights/yolov8 (1).pkl', 'rb') as file:
#        classification_model = pickle.load(file)
#    st.sidebar.success("Classification model loaded successfully!")
#except Exception as e:
#    st.sidebar.error(f"Error loading classification model: {str(e)}")

model = load_model(Path(DETECTION_MODEL))
if model is None:
    st.stop()  # Stop the app if the model fails to load

# Source selection
st.sidebar.markdown('<div class="sub-header">Source Selection</div>', unsafe_allow_html=True)
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)

# Main content area based on source selection
if source_radio == "Image":
    st.markdown('<div class="sub-header">Image Detection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        source_img = st.file_uploader(
            "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), 
            help="Upload an image containing waste items for classification."
        )
        if source_img is not None:
            st.info("Image uploaded! Click 'Detect Objects' to analyze.")
        else:
            st.info("Please upload an image or use the default example.")
    
    with col2:
        if source_img is None:
            try:
                default_image = PIL.Image.open(DEFAULT_IMAGE)
                st.image(default_image, caption="Example Image", use_container_width=True)
                st.caption("Default example image showing waste items.")
            except Exception as ex:
                st.error("Error loading default image.")
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button('Detect Objects', key="detect_objects"):
                with st.spinner('Detecting objects...'):
                    results = model.predict(uploaded_image, conf=confidence)
                    res_plotted = results[0].plot()[:, :, ::-1]
                    
                    st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        num_detections = len(boxes)
                        confidences = boxes.conf.tolist()
                        avg_confidence = sum(confidences) / num_detections
                        
                        st.success(f"Detection completed! Found {num_detections} object(s) with an average confidence of {avg_confidence:.2f}.")
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{num_detections}</div><div class="metric-label">Objects Detected</div></div>', unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_confidence:.2f}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
                        
                        with st.expander("Detection Details", expanded=True):
                            st.markdown("**Detected Objects:**")
                            for i, box in enumerate(boxes):
                                st.write(f"Object {i+1}: Class = {model.names[int(box.cls.item())]}, Confidence = {box.conf.item():.2f}")
                    else:
                        st.warning("No objects detected. Try lowering the confidence threshold.")

elif source_radio == "Webcam":
    st.markdown('<div class="sub-header">Webcam Detection</div>', unsafe_allow_html=True)
    play_webcam(confidence, model)

elif source_radio == "Video":
    st.markdown('<div class="sub-header">Video Detection</div>', unsafe_allow_html=True)
    handle_video_detection(confidence, model)
    # The YouTube detection functionality can be a separate button within the Video section if desired
    # but the primary flow is now via file upload.

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5C5C5C;">
    <p>Waste Classification App using YOLOv8 | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)
