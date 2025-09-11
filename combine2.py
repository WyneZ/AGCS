# Python In-built packages
from pathlib import Path
import PIL
import cv2
import av
# import pafy  # disable if not used
# import pickle

# External packages
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Setting page layout
st.set_page_config(
    page_title="AI Garbage Classification",
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
    .metric-card { background-color: #F0F4F7; padding: 15px; border-radius: 10px; text-align: center;
                   box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: 10px 0; border-left: 5px solid #004D73; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #37A67A; }
    .metric-label { font-size: 1.1rem; color: #5C5C5C; }
</style>
""", unsafe_allow_html=True)

# Default paths
DETECTION_MODEL = 'weights/best.pt'
DETECTION_MODEL2 = 'weights/yoloooo.pt'
DEFAULT_IMAGE = 'images/def.jfif'
SOURCES_LIST = ["Image", "Live", "Video"]

# Helper functions
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=False, tracker=None):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    results = model.predict(image, conf=conf, verbose=False)
    res_plotted = results[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)
    return results  # Return results for metrics

def display_detection_metrics(results, model):
    """Displays detection metrics including count, confidence, and details."""
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

# --- Webcam Snapshot Mode ---
def play_webcam_snapshot(conf, model):
    st.info("📸 Capture an image from your webcam")
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(cv2_img, conf=conf, verbose=False)
        res_plotted = results[0].plot()[:, :, ::-1]

        st.image(res_plotted, caption="Detected Objects", use_container_width=True)
        
        # Display detection metrics
        display_detection_metrics(results, model)

# --- Custom Video Processor for WebRTC ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perform detection
        results = self.model.predict(img, conf=self.conf, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Webcam Live Mode (streamlit-webrtc) ---
def play_webcam_live(conf):
    st.info("🎥 Live webcam detection (requires browser permission)")
    
    # Load the model only when needed
    model = load_model(Path(DETECTION_MODEL))
    if model is None:
        st.error("Failed to load model for live detection")
        return
    
    # Create the video processor with our model
    ctx = webrtc_streamer(
        key="yolo-live-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: YOLOVideoProcessor(model, conf),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.model = model
        ctx.video_processor.conf = conf

# --- Video File Detection ---
def handle_video_detection(conf):
    # Load the model only when needed
    model = load_model(Path(DETECTION_MODEL2))
    if model is None:
        st.error("Failed to load model for video detection")
        return
        
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
            
            # Create a placeholder for metrics
            metrics_placeholder = st.empty()
            all_detections = []
            
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    current_frame += 1
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                    results = _display_detected_frames(conf, model, st_frame, image)
                    
                    # Collect detection data for metrics
                    if results and results[0].boxes:
                        all_detections.extend(results[0].boxes)
                        
                else:
                    # Calculate and display final metrics
                    if all_detections:
                        num_detections = len(all_detections)
                        confidences = [box.conf.item() for box in all_detections]
                        avg_confidence = sum(confidences) / num_detections
                        
                        with metrics_placeholder.container():
                            st.success(f"Video processing finished! Found {num_detections} object(s) with an average confidence of {avg_confidence:.2f}.")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{num_detections}</div><div class="metric-label">Total Objects Detected</div></div>', unsafe_allow_html=True)
                            with cols[1]:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_confidence:.2f}</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
                            
                            with st.expander("Detection Summary", expanded=True):
                                st.markdown("**Detection Summary:**")
                                class_counts = {}
                                for box in all_detections:
                                    class_name = model.names[int(box.cls.item())]
                                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                
                                for class_name, count in class_counts.items():
                                    st.write(f"{class_name}: {count} detected")
                    
                    st.success("Video processing finished.")
                    vid_cap.release()
                    progress_bar.empty()
                    break
            if temp_path.exists():
                temp_path.unlink()

# --- Main ---
st.markdown('<h1 class="main-header">♻️ AI Garbage Classification</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This app uses <b>YOLOv8</b> to detect and classify waste.</p>
    <p>Choose Image, Live, or Video from the sidebar to start.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)
confidence = float(st.sidebar.slider("Confidence", 25, 100, 40)) / 100

st.sidebar.markdown('<div class="sub-header">Source</div>', unsafe_allow_html=True)
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)

if source_radio == "Image":
    st.markdown('<div class="sub-header">Image Detection</div>', unsafe_allow_html=True)
    
    # Load the model only when needed
    model = load_model(Path(DETECTION_MODEL2))
    if model is None:
        st.error("Failed to load model for image detection")
        st.stop()
        
    source_img = st.file_uploader("Upload an image...", type=("jpg","jpeg","png","bmp","webp"))
    if source_img is not None:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button('Detect Objects', key="detect_objects"):
            results = model.predict(uploaded_image, conf=confidence, verbose=False)
            res_plotted = results[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Objects', use_container_width=True)
            
            # Display detection metrics
            display_detection_metrics(results, model)

elif source_radio == "Live":
    st.markdown('<div class="sub-header">Webcam Live Stream</div>', unsafe_allow_html=True)
    # Use the original model for live webcam
    play_webcam_live(confidence)

elif source_radio == "Video":
    st.markdown('<div class="sub-header">Video Detection</div>', unsafe_allow_html=True)
    # Use the second model for video detection
    handle_video_detection(confidence)

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center; color:#5C5C5C;">AI Garbage Classification | Made by KuuKuu</div>', unsafe_allow_html=True)