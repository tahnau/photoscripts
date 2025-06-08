#!/usr/bin/env python3

"""
pip install torch torchvision transformers pillow
pip install flash-attn # optional, speedup

# Test run 1:
python3 image_captions.py /path/to/my/photos

# Test run 2
find /mnt/photos -type f \( -iname "*.jpg" -o -iname "*.png" \) | python3 image_captions.py

"""

import os
import sys
import hashlib
import json
import fcntl
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import logging
import traceback
import argparse

# Configuration
OUTPUT_FILE = "image_captions.txt"
STATE_FILE = ".caption_state.json"
LOCK_FILE = ".caption.lock"
# FALLBACK_INPUT_PATH will be used if no arguments and no stdin pipe are provided.
FALLBACK_INPUT_PATH = "/media/MY_PHOTO_ALBUM/" # Defaulting to a directory
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic", ".heif"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("caption_errors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_lock():
    """Prevent multiple instances"""
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except IOError:
        logging.error("Another instance is running or lock file is stale. If sure no other instance is running, delete .caption.lock")
        sys.exit(1)

def compute_hash(filepath):
    """Compute SHA256 hash of file"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_processed_files():
    """Load already processed file paths"""
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        processed.add(parts[0])
        except Exception as e:
            logging.warning(f"Could not properly read existing output file {OUTPUT_FILE}: {e}")
            pass
    return processed

def save_checkpoint(processed_paths):
    """Save checkpoint to resume later"""
    try:
        with open(STATE_FILE + '.tmp', 'w', encoding='utf-8') as f:
            json.dump({'paths': list(processed_paths)}, f)
        os.rename(STATE_FILE + '.tmp', STATE_FILE)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")


def sort_output_file():
    """Sort output file by path"""
    if not os.path.exists(OUTPUT_FILE):
        return
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines.sort()
        with open(OUTPUT_FILE + '.tmp', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        os.rename(OUTPUT_FILE + '.tmp', OUTPUT_FILE)
    except Exception as e:
        logging.error(f"Error sorting output file: {e}")


def collect_image_files_from_directory(directory_str: str):
    """Collect all image files from a directory (recursive)"""
    image_files = []
    path = Path(directory_str) # Assumes path is a valid, existing directory

    supported_extensions_lower = [ext.lower() for ext in SUPPORTED_EXTS]
    all_found_files = []
    for p_object in path.rglob('*'):
        if p_object.is_file() and p_object.suffix.lower() in supported_extensions_lower:
             all_found_files.append(str(p_object.resolve()))

    image_files = sorted(list(set(all_found_files))) # Unique and sort
    return image_files

def get_image_timestamp(image_path_str: str) -> str:
    """
    Attempts to get the image creation timestamp from EXIF data.
    Falls back to file modification time.
    Returns timestamp as an ISO 8601 formatted string or "Timestamp_Unavailable".
    """
    try:
        img = Image.open(image_path_str)
        exif_data = img._getexif()
        img.close()

        if exif_data:
            timestamp_str = None
            tag_priority = [36867, 36868, 306]
            for tag_id in tag_priority:
                if tag_id in exif_data:
                    raw_val = exif_data[tag_id]
                    if isinstance(raw_val, bytes): timestamp_str = raw_val.decode(errors='replace').strip()
                    elif isinstance(raw_val, str): timestamp_str = raw_val.strip()
                    if timestamp_str: break

            if timestamp_str:
                parsed_dt = None
                possible_formats = ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M:%S.%f']
                for fmt in possible_formats:
                    try:
                        parsed_dt = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError: continue

                if parsed_dt: return parsed_dt.isoformat()
                else:
                    try:
                        ts_iso_candidate = timestamp_str.replace(" ", "T").rstrip("Z")
                        if '.' in ts_iso_candidate: ts_iso_candidate = ts_iso_candidate.split('.')[0]
                        parsed_dt = datetime.fromisoformat(ts_iso_candidate)
                        return parsed_dt.isoformat()
                    except ValueError: logging.warning(f"Could not parse EXIF '{timestamp_str}' for {image_path_str}. Using mtime.")
            else: logging.info(f"No suitable EXIF tag for {image_path_str}. Using mtime.")
        else: logging.info(f"No EXIF data for {image_path_str}. Using mtime.")
    except UnidentifiedImageError: logging.warning(f"UnidentifiedImageError (EXIF) for {image_path_str}. Using mtime.")
    except FileNotFoundError: logging.error(f"File not found (EXIF): {image_path_str}."); return "Timestamp_Error_FileNotFound"
    except AttributeError: logging.warning(f"AttributeError (EXIF) for {image_path_str}. Using mtime.")
    except Exception as e: logging.warning(f"EXIF error for {image_path_str}: {type(e).__name__} - {e}. Using mtime.")

    try:
        mtime_unix = os.path.getmtime(image_path_str)
        return datetime.fromtimestamp(mtime_unix).isoformat()
    except Exception as e:
        logging.error(f"Could not get mtime for {image_path_str}: {e}")
        return "Timestamp_Unavailable"

def main(initial_paths_to_check: list[str]):
    lock_fd = None
    try:
        lock_fd = get_lock()

        all_potential_image_files = []
        for path_str_from_input in initial_paths_to_check:
            resolved_path = Path(path_str_from_input).resolve()
            logging.info(f"Processing input source: {resolved_path}")

            if not resolved_path.exists():
                logging.warning(f"Provided path does not exist and will be skipped: {resolved_path}")
                continue

            if resolved_path.is_file():
                if resolved_path.suffix.lower() in SUPPORTED_EXTS:
                    all_potential_image_files.append(str(resolved_path))
                else:
                    logging.warning(f"File '{resolved_path}' is not a supported image type. Supported: {', '.join(SUPPORTED_EXTS)}. Skipping.")
            elif resolved_path.is_dir():
                logging.info(f"Input '{resolved_path}' is a directory. Collecting images...")
                images_in_dir = collect_image_files_from_directory(str(resolved_path))
                if images_in_dir:
                    all_potential_image_files.extend(images_in_dir)
                    logging.info(f"Collected {len(images_in_dir)} images from {resolved_path}")
                else:
                    logging.info(f"No supported images found in directory {resolved_path}")
            else:
                logging.warning(f"Path '{resolved_path}' is neither a valid file nor a directory. Skipping.")

        # Consolidate all collected image file paths
        files_to_consider = sorted(list(set(all_potential_image_files)))

        if not files_to_consider:
            logging.info("No supported image files found from any input source after filtering.")
            return

        logging.info(f"Found {len(files_to_consider)} unique image file(s) to consider for processing across all inputs.")

        processed_paths_set = load_processed_files()
        logging.info(f"Found {len(processed_paths_set)} already processed file paths in {OUTPUT_FILE}")

        new_files = [f for f in files_to_consider if f not in processed_paths_set]

        if not new_files:
            logging.info("All found image(s) have already been processed or no new images to process.")
            return

        logging.info(f"Processing {len(new_files)} new image(s).")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_available = torch.cuda.is_available()
        logging.info(f"Using device: {device} (GPU: {gpu_available})")

        model_name = "microsoft/Florence-2-large"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        model_dtype = torch.float16 if gpu_available and torch.cuda.is_bf16_supported() else torch.float32
        if device.type == 'cpu': model_dtype = torch.float32
        logging.info(f"Loading model with dtype: {model_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=model_dtype
        ).eval().to(device)

        processed_count_this_run = 0
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
            for img_path_str in new_files:
                image_timestamp_str = "Timestamp_Not_Processed"
                try:
                    image_timestamp_str = get_image_timestamp(img_path_str)
                    task_prompt = "<MORE_DETAILED_CAPTION>"

                    image = Image.open(img_path_str).convert('RGB') # Open image after timestamp attempt
                    file_hash = compute_hash(img_path_str)

                    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model_dtype and device.type == 'cuda':
                         inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024, num_beams=3, early_stopping=True
                    )
                    generated_text_list = processor.batch_decode(generated_ids, skip_special_tokens=False)
                    generated_text = generated_text_list[0]

                    parsed_answer = processor.post_process_generation(
                        generated_text, task=task_prompt, image_size=(image.width, image.height)
                    )
                    caption = parsed_answer.get(task_prompt)
                    if caption is None:
                        logging.warning(f"Caption not found in model output for {img_path_str}. Raw: {generated_text}")
                        caption = "Error: Caption not found."
                    elif isinstance(caption, list): caption = " ".join(caption)

                    caption_cleaned = caption.replace('\n', ' ').replace('\t', ' ').strip()
                    out_file.write(f"{img_path_str}\t{image_timestamp_str}\t{file_hash}\t{caption_cleaned}\n")
                    out_file.flush()

                    processed_paths_set.add(img_path_str) # Add to the set of all processed paths
                    processed_count_this_run += 1
                    logging.info(f"Processed ({processed_count_this_run}/{len(new_files)}): {img_path_str} (TS: {image_timestamp_str}) -> {caption_cleaned[:60]}...")

                    if processed_count_this_run > 0 and processed_count_this_run % 10 == 0:
                        save_checkpoint(list(processed_paths_set)) # Save the complete set
                        logging.info(f"Checkpoint saved. Processed {processed_count_this_run} images in this run.")

                except FileNotFoundError: logging.error(f"Image file not found during processing: {img_path_str}")
                except UnidentifiedImageError: logging.error(f"Cannot identify image (corrupted/unsupported): {img_path_str}")
                except RuntimeError as e: logging.error(f"Runtime error processing {img_path_str}: {e}"); logging.error(traceback.format_exc())
                except Exception as e: logging.error(f"Generic error processing {img_path_str}: {e}"); logging.error(traceback.format_exc())

        save_checkpoint(list(processed_paths_set)) # Final checkpoint for this run
        sort_output_file()
        logging.info(f"Successfully processed {processed_count_this_run} new image(s) in this run.")

    except Exception as e:
        logging.critical(f"A critical error occurred in main: {e}")
        logging.critical(traceback.format_exc())
    finally:
        if lock_fd:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            try:
                if os.path.exists(LOCK_FILE): os.unlink(LOCK_FILE)
            except OSError as e:
                logging.warning(f"Could not remove lock file {LOCK_FILE}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate captions for images. Accepts image file paths or directory paths as arguments and/or via stdin."
    )
    parser.add_argument(
        "input_paths",
        metavar="PATH",
        nargs='*', # Zero or more paths
        help=(
            "One or more paths to image files or directories. "
            "If data is piped via stdin, paths are also read from stdin. "
            f"If no arguments are provided and stdin is not piped, defaults to: '{FALLBACK_INPUT_PATH}'"
        )
    )
    args = parser.parse_args()

    cmd_line_paths = args.input_paths

    stdin_paths = []
    if not sys.stdin.isatty(): # True if stdin is piped or redirected from a file
        logging.info("Reading input paths from stdin...")
        for line in sys.stdin:
            path = line.strip()
            if path: # Ignore empty lines
                stdin_paths.append(path)
        if stdin_paths:
            logging.info(f"Read {len(stdin_paths)} path(s) from stdin.")
        else:
            logging.info("Stdin was piped, but no paths were read.")

    initial_paths_to_check = []
    if cmd_line_paths:
        initial_paths_to_check.extend(cmd_line_paths)
    if stdin_paths:
        initial_paths_to_check.extend(stdin_paths)

    if not initial_paths_to_check:
        logging.info(f"No input paths from arguments or stdin. Using fallback: {FALLBACK_INPUT_PATH}")
        initial_paths_to_check.append(FALLBACK_INPUT_PATH)

    logging.info(f"Initial list of {len(initial_paths_to_check)} input sources to investigate.")

    try:
        import flash_attn # Check for flash_attn before main model loading
    except ImportError:
        logging.warning("flash_attn not found. Model loading might use a fallback or require it.")
        pass

    main(initial_paths_to_check)
