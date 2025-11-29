import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
import json
from io import BytesIO
import zipfile
from datetime import datetime
import gdown
import torch
from ultralytics.nn.tasks import DetectionModel
import keras

# Classification and Detection Classes
CLASSIFICATION_CLASSES = ["non-venomous", "venomous"]
DETECTION_CLASSES = ["snake"]

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Snake Detection & Classification",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .venomous {
        color: #d62728;
        font-weight: bold;
    }
    .non-venomous {
        color: #2ca02c;
        font-weight: bold;
    }
    .classification-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
    }
    .venomous-box {
        background-color: #ffebee;
        border: 3px solid #d62728;
    }
    .non-venomous-box {
        background-color: #e8f5e9;
        border: 3px solid #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Models (with caching)
# ---------------------------
@st.cache_resource
def load_models():
    """Download and load YOLO detection and TensorFlow classifier models"""
    import os
    import gdown
    import torch
    from ultralytics import YOLO
    import tensorflow as tf

    # Google Drive file IDs
    YOLO_ID = "1DH5zyX4jBNA3aLPjiwtA0Gh_HEm5z9cv"
    CLASSIFIER_ID = "17tXUZkDWK4a2ia7DbNhWhS2k4_DbtiYc"

    # Create directories if missing
    os.makedirs("weights", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Download YOLO model if missing
    yolo_path = "weights/best.pt"
    if not os.path.exists(yolo_path):
        st.info("🔽 Downloading YOLO model... (one-time setup, ~200MB)")
        try:
            gdown.download(f"https://drive.google.com/uc?id={YOLO_ID}", yolo_path, quiet=False)
            st.success("✅ YOLO model downloaded!")
        except Exception as e:
            st.error(f"❌ Error downloading YOLO model: {e}")
            return None, None

    # Download classifier if missing
    classifier_path = "models/snake_venom_classifier_effnetv2L.h5"
    if not os.path.exists(classifier_path):
        st.info("🔽 Downloading classifier model... (one-time setup, ~750MB, may take 5-10 minutes)")
        try:
            gdown.download(f"https://drive.google.com/uc?id={CLASSIFIER_ID}", classifier_path, quiet=False)
            st.success("✅ Classifier model downloaded!")
        except Exception as e:
            st.error(f"❌ Error downloading classifier model: {e}")
            return None, None

    # Load YOLO model
    try:
        st.info("🔄 Loading YOLO model...")
        
        # Handle PyTorch safe globals for newer versions
        try:
            from ultralytics.nn.tasks import DetectionModel
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([DetectionModel])
        except:
            pass  # Older PyTorch versions don't need this
        
        # Load YOLO
        yolo_model = YOLO(yolo_path)
        st.success("✅ YOLO model loaded!")
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        return None, None

    # Load TensorFlow classifier with compatibility fix
    try:
        st.info("🔄 Loading classifier model...")
        
        # Fix for old Keras models with batch_shape parameter
        from tensorflow.keras import layers
        
        class CompatibleInputLayer(layers.InputLayer):
            def __init__(self, batch_shape=None, input_shape=None, **kwargs):
                # Handle old batch_shape parameter
                if batch_shape is not None and input_shape is None:
                    input_shape = batch_shape[1:]
                # Remove batch_shape from kwargs
                kwargs.pop('batch_shape', None)
                super().__init__(input_shape=input_shape, **kwargs)
        
        # Custom objects for loading
        custom_objects = {
            'InputLayer': CompatibleInputLayer,
        }
        
        # Load with custom objects and without compilation
        classifier_model = tf.keras.models.load_model(
            classifier_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Compile with current optimizer
        classifier_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Classifier model loaded!")
    except Exception as e:
        st.error(f"❌ Error loading classifier model: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return yolo_model, None

    return yolo_model, classifier_model
# ---------------------------
# Helper Functions
# ---------------------------
def classify_full_image(image, classifier_model):
    """Classify the full image using EfficientNetV2L preprocessing"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Ensure RGB format
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size (384x384 for EfficientNetV2L)
    img_resized = cv2.resize(img, (384, 384))
    
    # CRITICAL: Use EfficientNetV2 preprocessing (same as training)
    # This applies the correct normalization that the model expects
    img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Predict
    preds = classifier_model.predict(img_batch, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][class_idx])
    
    return CLASSIFICATION_CLASSES[class_idx], confidence

def detect_snakes(image, yolo_model, conf_threshold=0.25):
    """Detect snakes in image using YOLO"""
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Ensure RGB format for YOLO
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert to BGR for YOLO
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection
    results = yolo_model.predict(img_bgr, conf=conf_threshold, verbose=False)
    detections = []
    
    for result in results:
        if len(result.boxes) == 0:
            continue
            
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for i, cls_idx in enumerate(classes):
            class_name = DETECTION_CLASSES[int(cls_idx)]
            if class_name != "snake":
                continue
            
            det_conf = float(confidences[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": det_conf
            })
    
    return detections

def draw_detections(image, detections, classification_label, classification_conf):
    """Draw bounding boxes on image based on classification result"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Ensure RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Determine color based on classification
    color = (255, 0, 0) if classification_label == "venomous" else (0, 255, 0)  # Red/Green in RGB
    
    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label text
        label_text = f"{classification_label} {classification_conf:.2%}"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(img, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def classify_then_detect(image, yolo_model, classifier_model, conf_threshold=0.25):
    """
    New pipeline: Classify full image first, then detect and draw boxes
    """
    # Step 1: Classify the full image
    classification_label, classification_conf = classify_full_image(image, classifier_model)
    
    # Step 2: Detect snakes in the image
    detections = detect_snakes(image, yolo_model, conf_threshold)
    
    # Step 3: Draw detections with classification result
    if detections:
        processed_img = draw_detections(image, detections, classification_label, classification_conf)
    else:
        # No detections, return original image
        if isinstance(image, Image.Image):
            processed_img = np.array(image)
        else:
            processed_img = image.copy()
    
    return processed_img, classification_label, classification_conf, detections

# ---------------------------
# Download Helper Functions
# ---------------------------
def create_result_json(filename, classification_label, classification_conf, detections):
    """Create JSON with all detection and classification details"""
    result = {
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classification": {
            "label": classification_label,
            "confidence": float(classification_conf)
        },
        "detections": []
    }
    
    for idx, det in enumerate(detections, 1):
        result["detections"].append({
            "snake_number": idx,
            "bounding_box": {
                "x1": int(det['bbox'][0]),
                "y1": int(det['bbox'][1]),
                "x2": int(det['bbox'][2]),
                "y2": int(det['bbox'][3])
            },
            "detection_confidence": float(det['detection_confidence'])
        })
    
    result["total_snakes_detected"] = len(detections)
    
    return result

def create_result_text(filename, classification_label, classification_conf, detections):
    """Create human-readable text report"""
    text = f"Snake Detection & Classification Report\n"
    text += f"{'=' * 50}\n\n"
    text += f"Filename: {filename}\n"
    text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    text += f"CLASSIFICATION RESULT:\n"
    text += f"  Label: {classification_label.upper()}\n"
    text += f"  Confidence: {classification_conf:.2%}\n\n"
    text += f"DETECTION RESULTS:\n"
    text += f"  Total Snakes Detected: {len(detections)}\n\n"
    
    if detections:
        for idx, det in enumerate(detections, 1):
            text += f"  Snake #{idx}:\n"
            text += f"    Bounding Box: [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]\n"
            text += f"    Detection Confidence: {det['detection_confidence']:.2%}\n\n"
    else:
        text += "  No snakes detected by YOLO.\n\n"
    
    return text

def create_download_package(processed_img, filename, classification_label, classification_conf, detections):
    """Create a ZIP file with image, JSON, and text report"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add processed image
        img_buffer = BytesIO()
        Image.fromarray(processed_img).save(img_buffer, format='PNG')
        zip_file.writestr(f"{filename}_processed.png", img_buffer.getvalue())
        
        # Add JSON report
        json_data = create_result_json(filename, classification_label, classification_conf, detections)
        zip_file.writestr(f"{filename}_report.json", json.dumps(json_data, indent=2))
        
        # Add text report
        text_data = create_result_text(filename, classification_label, classification_conf, detections)
        zip_file.writestr(f"{filename}_report.txt", text_data)
    
    zip_buffer.seek(0)
    return zip_buffer

def create_batch_download_package(results_list):
    """Create a ZIP file with all batch results"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create summary JSON for all images
        batch_summary = {
            "batch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(results_list),
            "results": []
        }
        
        for result in results_list:
            filename = result['filename']
            processed_img = result['processed_img']
            classification_label = result['classification_label']
            classification_conf = result['classification_conf']
            detections = result['detections']
            
            # Add processed image
            img_buffer = BytesIO()
            Image.fromarray(processed_img).save(img_buffer, format='PNG')
            zip_file.writestr(f"images/{filename}_processed.png", img_buffer.getvalue())
            
            # Add individual JSON
            json_data = create_result_json(filename, classification_label, classification_conf, detections)
            zip_file.writestr(f"reports/{filename}_report.json", json.dumps(json_data, indent=2))
            
            # Add individual text report
            text_data = create_result_text(filename, classification_label, classification_conf, detections)
            zip_file.writestr(f"reports/{filename}_report.txt", text_data)
            
            # Add to batch summary
            batch_summary["results"].append(json_data)
        
        # Add batch summary
        zip_file.writestr("batch_summary.json", json.dumps(batch_summary, indent=2))
        
        # Create overall summary text
        summary_text = f"Batch Processing Summary\n"
        summary_text += f"{'=' * 50}\n\n"
        summary_text += f"Total Images Processed: {len(results_list)}\n"
        summary_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        venomous_count = sum(1 for r in results_list if r['classification_label'] == 'venomous')
        non_venomous_count = len(results_list) - venomous_count
        
        summary_text += f"Classification Summary:\n"
        summary_text += f"  Venomous: {venomous_count}\n"
        summary_text += f"  Non-venomous: {non_venomous_count}\n\n"
        
        summary_text += f"Detailed Results:\n"
        for result in results_list:
            summary_text += f"\n  {result['filename']}:\n"
            summary_text += f"    Classification: {result['classification_label'].upper()} ({result['classification_conf']:.2%})\n"
            summary_text += f"    Detections: {len(result['detections'])} snake(s)\n"
        
        zip_file.writestr("batch_summary.txt", summary_text)
    
    zip_buffer.seek(0)
    return zip_buffer

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown('<h1 class="main-header">🐍 Snake Detection & Classification System</h1>', unsafe_allow_html=True)
    
    # Load models
    yolo_model, classifier_model = load_models()
    
    if yolo_model is None or classifier_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for YOLO to detect a snake"
    )
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["📸 Image Upload", "📁 Batch Upload", "📹 Webcam (Real-time)"],
        help="Choose how you want to detect snakes"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**New Pipeline:**\n"
        "1. **Classify** the full image as venomous/non-venomous\n"
        "2. **Detect** snake locations with YOLO\n"
        "3. **Draw** bounding boxes with classification result\n\n"
        "This approach uses the full image for better classification accuracy!"
    )
    
    # ---------------------------
    # Image Upload Mode
    # ---------------------------
    if mode == "📸 Image Upload":
        st.header("Upload a Single Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing a snake"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process image with new pipeline
            with st.spinner("Classifying and detecting snakes..."):
                processed_img, classification_label, classification_conf, detections = classify_then_detect(
                    image, yolo_model, classifier_model, conf_threshold
                )
            
            with col2:
                st.subheader("Detection Results")
                st.image(processed_img, use_container_width=True)
            
            # Display classification result prominently
            label_class = "venomous" if classification_label == "venomous" else "non-venomous"
            box_class = "venomous-box" if classification_label == "venomous" else "non-venomous-box"
            
            st.markdown(
                f'<div class="classification-box {box_class}">'
                f'<span class="{label_class}">Classification: {classification_label.upper()}</span><br>'
                f'Confidence: {classification_conf:.2%}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Display detection details
            if detections:
                st.success(f"✅ Found {len(detections)} snake(s) in the image!")
                
                for idx, det in enumerate(detections, 1):
                    st.markdown(
                        f"**Snake {idx}:** Detection Confidence: {det['detection_confidence']:.2%}",
                    )
            else:
                st.warning("⚠️ No snakes detected by YOLO, but image was classified.")
            
            # Download Section
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download processed image
                img_buffer = BytesIO()
                Image.fromarray(processed_img).save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📷 Download Image",
                    data=img_buffer,
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed.png",
                    mime="image/png"
                )
            
            with col2:
                # Download JSON report
                json_data = create_result_json(
                    uploaded_file.name, 
                    classification_label, 
                    classification_conf, 
                    detections
                )
                
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"{uploaded_file.name.split('.')[0]}_report.json",
                    mime="application/json"
                )
            
            with col3:
                # Download complete package (ZIP)
                zip_buffer = create_download_package(
                    processed_img,
                    uploaded_file.name.split('.')[0],
                    classification_label,
                    classification_conf,
                    detections
                )
                
                st.download_button(
                    label="📦 Download Package",
                    data=zip_buffer,
                    file_name=f"{uploaded_file.name.split('.')[0]}_results.zip",
                    mime="application/zip",
                    help="Downloads a ZIP with image, JSON, and text report"
                )
    
    # ---------------------------
    # Batch Upload Mode
    # ---------------------------
    elif mode == "📁 Batch Upload":
        st.header("Upload Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload multiple images to process in batch"
        )
        
        if uploaded_files:
            st.info(f"Processing {len(uploaded_files)} image(s)...")
            
            # Store all results for batch download
            batch_results = []
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files, 1):
                st.markdown(f"### Image {idx}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                
                with st.spinner(f"Processing image {idx}..."):
                    processed_img, classification_label, classification_conf, detections = classify_then_detect(
                        image, yolo_model, classifier_model, conf_threshold
                    )
                
                with col2:
                    st.image(processed_img, caption="Processed", use_container_width=True)
                
                # Classification result
                label_class = "venomous" if classification_label == "venomous" else "non-venomous"
                st.markdown(
                    f"**Classification:** <span class='{label_class}'>{classification_label.upper()}</span> "
                    f"({classification_conf:.2%})",
                    unsafe_allow_html=True
                )
                
                # Detection result
                if detections:
                    st.success(f"Found {len(detections)} snake(s)")
                else:
                    st.warning("No snakes detected by YOLO")
                
                # Store result for batch download
                batch_results.append({
                    'filename': uploaded_file.name,
                    'processed_img': processed_img,
                    'classification_label': classification_label,
                    'classification_conf': classification_conf,
                    'detections': detections
                })
                
                st.markdown("---")
            
            # Batch download section
            st.markdown("---")
            st.subheader("📥 Download All Results")
            st.info(f"Download all {len(batch_results)} processed images with reports in a single ZIP file")
            
            # Create batch download package
            batch_zip = create_batch_download_package(batch_results)
            
            st.download_button(
                label=f"📦 Download All Results ({len(batch_results)} images)",
                data=batch_zip,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="Downloads a ZIP with all processed images, individual reports, and a batch summary"
            )
            
            # Show batch statistics
            with st.expander("📊 Batch Statistics"):
                venomous_count = sum(1 for r in batch_results if r['classification_label'] == 'venomous')
                non_venomous_count = len(batch_results) - venomous_count
                total_detections = sum(len(r['detections']) for r in batch_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Venomous", venomous_count)
                with col2:
                    st.metric("Non-venomous", non_venomous_count)
                with col3:
                    st.metric("Total Detections", total_detections)
    
    # ---------------------------
    # Webcam Mode
    # ---------------------------
    elif mode == "📹 Webcam (Real-time)":
        st.header("Real-time Webcam Detection")
        st.warning("⚠️ This feature requires camera permissions and may not work on all deployments.")
        
        run_webcam = st.checkbox("Start Webcam")
        
        if run_webcam:
            stframe = st.empty()
            classification_placeholder = st.empty()
            camera = cv2.VideoCapture(0)
            
            stop_button = st.button("Stop Webcam")
            
            while run_webcam and not stop_button:
                ret, frame = camera.read()
                
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with new pipeline
                processed_frame, classification_label, classification_conf, detections = classify_then_detect(
                    frame_rgb, yolo_model, classifier_model, conf_threshold
                )
                
                # Display processed frame
                stframe.image(processed_frame, channels="RGB", use_container_width=True)
                
                # Display classification
                label_class = "venomous" if classification_label == "venomous" else "non-venomous"
                classification_placeholder.markdown(
                    f"**Current Classification:** <span class='{label_class}'>{classification_label.upper()}</span> "
                    f"({classification_conf:.2%})",
                    unsafe_allow_html=True
                )
            
            camera.release()

if __name__ == "__main__":
    main()
