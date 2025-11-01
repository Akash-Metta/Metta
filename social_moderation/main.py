"""
Smart Content Moderation Pipeline
Face Blur + Hate Speech Detection + Blood/NSFW Detection
"""

import cv2
import argparse
from modules.face_blur_p import FaceBlurrer
from modules.text_blur_p import TextBlurrer
from modules.nsfw_blur import NSFWBlurrer
import os
import sys
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_blur_strength(blur_strength):
    """Ensure blur strength is odd and reasonable"""
    if blur_strength % 2 == 0:
        blur_strength += 1
    return min(max(blur_strength, 3), 151)

def get_media_type(input_path, media_type_arg):
    """Determine media type"""
    if media_type_arg != 'auto':
        return media_type_arg
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    ext = Path(input_path).suffix.lower()
    
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    
    return 'image'

def process_image(input_path, output_path, blur_strength, confidence, exclude_center, blur_text_p, 
                  nsfw_blur, nsfw_blur_type, selective_nsfw, nsfw_threshold=0.7, 
                  violence_threshold=0.6, blood_threshold=0.5):
    """Process image with face blur + hate speech + blood detection"""
    
    try:
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")

        blur_strength = validate_blur_strength(blur_strength)
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), 'config_p.yaml')
        config_path = os.path.abspath(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update blur strength in config
        config['blur']['face']['gaussian_min_kernel'] = blur_strength
        config['blur']['face']['gaussian_max_kernel'] = blur_strength
        config['blur']['face']['mosaic_block_size'] = max(2, blur_strength // 6)
        config['blur']['text']['gaussian_min_kernel'] = blur_strength
        config['blur']['text']['gaussian_max_kernel'] = blur_strength
        config['blur']['text']['mosaic_block_size'] = max(2, blur_strength // 6)

        # Initialize face detector and blurrer
        from detectors.yolov8_face import YOLOv8Face
        face_detector = YOLOv8Face(conf=confidence)
        face_blurrer = FaceBlurrer(face_detector, config)
        
        logger.info(f"Processing: {Path(input_path).name}")
        logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
        logger.info(f"Blur strength: {blur_strength}, Confidence: {confidence}")
        logger.info(f"Text blur: {blur_text_p}, NSFW blur: {nsfw_blur}")
        
        output = image.copy()
        
        # ============================================================
        # STEP 1: BLOOD/NSFW/VIOLENCE DETECTION
        # ============================================================
        if nsfw_blur:
            logger.info("=" * 60)
            logger.info("STEP 1: BLOOD/NSFW/VIOLENCE DETECTION & BLURRING")
            logger.info("=" * 60)
            
            nsfw_blurrer = NSFWBlurrer(
                blur_strength=(blur_strength, blur_strength),
                blur_type=nsfw_blur_type,
                blood_threshold=blood_threshold
            )
            
            try:
                result = nsfw_blurrer.blur_unsafe_content(output, add_warning=True)
                output = result['image']
                if result['analysis']:
                    logger.info(f"✓ Analysis: {result['analysis']['flags']}")
            except Exception as e:
                logger.warning(f"NSFW blur failed: {e}", exc_info=True)
        
        # ============================================================
        # STEP 2: HATE SPEECH/OFFENSIVE TEXT DETECTION
        # ============================================================
        if blur_text_p:
            logger.info("=" * 60)
            logger.info("STEP 2: HATE SPEECH/OFFENSIVE TEXT DETECTION & BLURRING")
            logger.info("=" * 60)
            
            try:
                text_blurrer = TextBlurrer(blur_strength=(blur_strength, blur_strength))
                logger.info("TextBlurrer initialized, attempting to blur text...")
                
                output = text_blurrer.blur_hate_text(output, confidence)
                logger.info("✓ Text blurred using blur_hate_text")
            except Exception as e:
                logger.warning(f"Text blur failed: {e}", exc_info=True)
        
        # ============================================================
        # STEP 3: FACE DETECTION & BLURRING
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 3: FACE DETECTION & BLURRING")
        logger.info("=" * 60)
        
        try:
            output = face_blurrer.blur_faces(output)
            logger.info("✓ Faces blurred successfully")
        except Exception as e:
            logger.warning(f"Face blur failed: {e}", exc_info=True)
        
        # Save result
        cv2.imwrite(output_path, output)
        logger.info("=" * 60)
        logger.info(f"✓ Successfully saved blurred image to: {output_path}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"✗ Error processing image: {e}", exc_info=True)
        return False

def process_video(input_path, output_path, blur_strength, confidence, exclude_center, blur_text, 
                  nsfw_blur, nsfw_blur_type, selective_nsfw, nsfw_threshold=0.7, 
                  violence_threshold=0.6, blood_threshold=0.5):
    """Process video frame by frame"""
    
    cap = None
    out = None
    
    try:
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            raise ValueError("Invalid video properties")
        
        blur_strength = validate_blur_strength(blur_strength)
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), 'config_p.yaml')
        config_path = os.path.abspath(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['blur']['face']['gaussian_min_kernel'] = blur_strength
        config['blur']['face']['gaussian_max_kernel'] = blur_strength
        config['blur']['face']['mosaic_block_size'] = max(2, blur_strength // 6)
        config['blur']['text']['gaussian_min_kernel'] = blur_strength
        config['blur']['text']['gaussian_max_kernel'] = blur_strength
        config['blur']['text']['mosaic_block_size'] = max(2, blur_strength // 6)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Initialize blurrers
        from detectors.yolov8_face import YOLOv8Face
        face_detector = YOLOv8Face(conf=confidence)
        face_blurrer = FaceBlurrer(face_detector, config)
        
        text_blurrer = None
        if blur_text:
            text_blurrer = TextBlurrer(blur_strength=(blur_strength, blur_strength))
        
        nsfw_blurrer = None
        if nsfw_blur:
            nsfw_blurrer = NSFWBlurrer(
                blur_strength=(blur_strength, blur_strength),
                blur_type=nsfw_blur_type,
                blood_threshold=blood_threshold
            )
        
        logger.info("=" * 60)
        logger.info(f"Processing video: {Path(input_path).name}")
        logger.info(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        logger.info(f"Blur strength: {blur_strength}, Confidence: {confidence}")
        logger.info(f"Text blur: {'enabled' if blur_text else 'disabled'}")
        logger.info(f"NSFW blur: {'enabled' if nsfw_blur else 'disabled'}")
        logger.info("=" * 60)
        
        # Process frames
        frame_count = 0
        text_blur_interval = 15
        nsfw_check_interval = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()
            
            # Apply NSFW/blood blur
            if nsfw_blur and nsfw_blurrer:
                if frame_count % nsfw_check_interval == 0:
                    try:
                        result = nsfw_blurrer.blur_unsafe_content(processed_frame, add_warning=False)
                        processed_frame = result['image']
                    except:
                        pass
            
            # Apply text blur
            if blur_text and text_blurrer and frame_count % text_blur_interval == 0:
                try:
                    processed_frame = text_blurrer.blur_hate_text(processed_frame, confidence)
                except:
                    pass
            
            # Apply face blur
            try:
                processed_frame = face_blurrer.blur_faces(processed_frame)
            except:
                pass
            
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        logger.info("=" * 60)
        logger.info(f"✓ Successfully processed {frame_count} frames")
        logger.info(f"✓ Saved blurred video to: {output_path}")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"✗ Error processing video: {e}", exc_info=True)
        return False
        
    finally:
        if cap:
            cap.release()
        if out:
            out.release()

def main():
    parser = argparse.ArgumentParser(
        description='Smart Content Moderation - Blur Faces, Hate Speech, Blood/NSFW'
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--media-type', '-t', choices=['image', 'video', 'auto'], default='auto')
    parser.add_argument('--blur-strength', '-b', type=int, default=51, help='Blur strength (3-151)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Detection confidence (0.1-1.0)')
    parser.add_argument('--exclude-center', action='store_true', help='Exclude center from face blur')
    parser.add_argument('--blur-text', action='store_true', help='Enable hate speech text blurring')
    parser.add_argument('--nsfw-blur', action='store_true', help='Enable blood/NSFW blurring')
    parser.add_argument('--nsfw-blur-type', choices=['gaussian', 'pixelate', 'mosaic', 'black'], default='gaussian')
    parser.add_argument('--nsfw-threshold', type=float, default=0.7, help='NSFW threshold')
    parser.add_argument('--violence-threshold', type=float, default=0.6, help='Violence threshold')
    parser.add_argument('--blood-threshold', type=float, default=0.5, help='Blood detection threshold')
    parser.add_argument('--selective-nsfw', action='store_true', help='Selective NSFW blur')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file '{args.input}' does not exist")
        
        media_type = get_media_type(args.input, args.media_type)
        logger.info(f"Detected media type: {media_type}")
        
        if not args.output:
            input_path = Path(args.input)
            output_dir = Path("data/output/images" if media_type == 'image' else "data/output/videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output = str(output_dir / f"{input_path.stem}_blurred{input_path.suffix}")
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        if media_type == 'image':
            success = process_image(
                args.input, args.output, args.blur_strength, 
                args.confidence, args.exclude_center, args.blur_text,
                args.nsfw_blur, args.nsfw_blur_type, args.selective_nsfw,
                args.nsfw_threshold, args.violence_threshold, args.blood_threshold
            )
        else:
            success = process_video(
                args.input, args.output, args.blur_strength, 
                args.confidence, args.exclude_center, args.blur_text,
                args.nsfw_blur, args.nsfw_blur_type, args.selective_nsfw,
                args.nsfw_threshold, args.violence_threshold, args.blood_threshold
            )
        
        sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
