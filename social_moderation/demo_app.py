import streamlit as st
from social_moderation.pipeline.processor import Processor
import tempfile

st.title("Social Moderation Demo")
video = st.file_uploader("Upload a video to test", type=["mp4", "mov", "avi"])
if st.button("Run Moderation"):
    proc = Processor(config_path="config.yaml")
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with open(tmp_in, "wb") as f:
        f.write(video.getvalue())
    proc.process_video(tmp_in, tmp_out)
    st.video(tmp_out)
    st.success("✅ Done! You can download the processed video below:")
    with open(tmp_out, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="blurred_output.mp4")
