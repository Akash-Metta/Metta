"""
Streamlit Demo App for HARI Content Moderation System
Interactive UI for testing moderation on uploaded videos/images
"""

import streamlit as st
import cv2
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HARI Content Moderation",
    page_icon="🛡️",
    layout="wide"
)

def main():
    st.title("🛡️ HARI Content Moderation System")
    st.markdown(
        "Upload videos or images to automatically blur faces and toxic text. "
        "Powered by YOLOv8, EasyOCR, and Detoxify."
    )
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Moderation toggles
    enable_face_blur = st.sidebar.checkbox("🙂 Blur Faces", value=True)
    enable_text_blur = st.sidebar.checkbox("💬 Blur Toxic Text", value=True)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        blur_method = st.selectbox("Blur Method", ["gaussian", "mosaic", "pixelate"])
        frame_skip = st.slider("Frame Skip (process every Nth frame)", 1, 10, 2)
        toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.7, 0.05)
        motion_smoothing = st.checkbox("Motion Smoothing (video)", value=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Video/Image",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        suffix = Path(uploaded_file.name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name
        
        # Display original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original")
            
            if uploaded_file.type.startswith('image'):
                st.image(input_path, use_container_width=True)
            else:
                st.video(input_path)
        
        # Process button
        if st.button("🚀 Run Moderation", type="primary"):
            with col2:
                st.subheader("📤 Moderated")
                
                with st.spinner("Processing... This may take a moment."):
                    try:
                        # Initialize processor
                        from social_moderation.pipeline.processor import Processor
                        
                        processor = Processor(config_path="config.yaml")
                        
                        # Update config with UI settings
                        processor.config["blur"]["face"]["method"] = blur_method
                        processor.config["blur"]["text"]["method"] = blur_method
                        processor.config["system"]["frame_skip"] = frame_skip
                        processor.config["toxicity"]["threshold"] = toxicity_threshold
                        processor.config["face_detector"]["motion_smoothing"]["enabled"] = motion_smoothing
                        
                        # Process
                        output_path = input_path.replace(suffix, f"_moderated{suffix}")
                        
                        if uploaded_file.type.startswith('image'):
                            # Image processing
                            image = cv2.imread(input_path)
                            
                            if enable_face_blur:
                                image = processor.face_blurrer.blur_faces(image)
                            
                            if enable_text_blur:
                                image = processor.text_blurrer.blur_toxic_text(image)
                            
                            cv2.imwrite(output_path, image)
                            st.image(output_path, use_container_width=True)
                        
                        else:
                            # Video processing
                            processor.process_video(input_path, output_path)
                            st.video(output_path)
                        
                        st.success("✅ Processing complete!")
                        
                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="💾 Download Moderated File",
                                data=f,
                                file_name=f"moderated_{uploaded_file.name}",
                                mime=uploaded_file.type
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        logger.exception("Processing error")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info(
        "HARI Content Moderation System uses state-of-the-art AI models to "
        "automatically detect and blur faces and toxic text in media content."
    )

if __name__ == '__main__':
    main()
