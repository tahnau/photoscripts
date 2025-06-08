#!/usr/bin/env python3

"""
Photo Organizer Script

This script scans a folder of photos and videos - such as phone photo sync folder -, extracts creation timestamps (via EXIF metadata or file modification time), and organizes the media into date-based directories under an output path. It also computes MD5 hashes to detect exact duplicates—moving those to a separate duplicates directory—and supports a simulation mode to preview file moves without making changes.
"""

import os
import shutil
import subprocess
import hashlib
from datetime import datetime
import logging
from pathlib import Path

# Directory for your photos
PHOTOS_DIR    = Path("/media/my_images/000_INPUT")
DUPLICATE_DIR = Path("/media/my_images/000_DUPLICATES")
OUTPUT_DIR    = Path("/media/my_images/000_OUTPUT")
SIMULATE = False  # Set to False to actually move the files

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if required tools are installed
def check_tool_installed(tool):
    if shutil.which(tool) is None:
        logging.error(f"{tool} is required but it's not installed. Aborting.")
        exit(1)

check_tool_installed('exiftool')
check_tool_installed('ffmpeg')

# Function to get the timestamp of an image or video
def get_timestamp(file):
    ext = file.suffix.lower()
    timestamp = None

    try:
        if ext in ['.jpg', '.jpeg', '.png', '.heic']:
            for tag in ['DateTimeOriginal', 'CreateDate', 'DateTimeDigitized']:
                result = subprocess.run(['exiftool', f'-{tag}', '-d', '%Y%m%d_%H%M%S', str(file)], capture_output=True, text=True)
                if result.stdout:
                    timestamp = result.stdout.split(': ')[1].strip()
                    if timestamp:
                        break
        elif ext in ['.mp4', '.mov']:
            result = subprocess.run(['ffmpeg', '-i', str(file)], capture_output=True, text=True, errors='ignore')
            for line in result.stderr.split('\n'):
                if 'creation_time' in line:
                    timestamp = line.split('creation_time   : ')[1].strip().replace('T', '_').replace('-', '').replace(':', '').split('.')[0]
                    break

        if timestamp:
            year = int(timestamp[:4])
            if year < 1990:
                timestamp = None
    except Exception as e:
        logging.error(f"Error getting timestamp for {file}: {e}")

    return timestamp

# Function to calculate the MD5 hash of a file
def calculate_md5(file_path, block_size=4096):
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(block_size), b''):
                md5.update(chunk)
    except Exception as e:
        logging.error(f"Error calculating MD5 for {file_path}: {e}")
        return None
    return md5.hexdigest()

# Function to handle duplicates
def handle_duplicates(target_file, source_file):
    source_file_hash = calculate_md5(source_file)
    if source_file_hash is None:
        logging.error(f"Skipping file {source_file} due to hash calculation error.")
        return None, False

    base = target_file.stem
    ext = target_file.suffix

    if target_file.exists():
        if calculate_md5(target_file) == source_file_hash:
            logging.info(f"Duplicate file found with same hash {source_file_hash}, skipping {target_file}")
            return None, True
    else:
        return target_file, False

    i = 1
    while target_file.with_name(f"{base}_img{i}{ext}").exists():
        if calculate_md5(target_file.with_name(f"{base}_img{i}{ext}")) == source_file_hash:
            logging.info(f"Duplicate file found with same hash {source_file_hash}, skipping {target_file.with_name(f'{base}_img{i}{ext}')}")
            return None, True
        i += 1

    return target_file.with_name(f"{base}_img{i}{ext}"), False


# Function to process images
def process_images(input_dir, output_dir):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.mp4', '.mov', '.heic', '.avi'):
                logging.info(f"Processing file: {file_path}")
                timestamp = get_timestamp(file_path)

                # If no EXIF timestamp is found, use file's modification time
                if not timestamp:
                    mod_time = file_path.stat().st_mtime
                    timestamp = datetime.fromtimestamp(mod_time).strftime('%Y%m%d_%H%M%S')
                    year_month = datetime.fromtimestamp(mod_time).strftime('%Y-%m')
                    target_dir = output_dir / f"{year_month}/{year_month}_telegram"
                else:
                    year_month = datetime.strptime(timestamp.split('_')[0], "%Y%m%d").strftime("%Y-%m")
                    target_dir = output_dir / year_month

                target_file = target_dir / f"{timestamp}{file_path.suffix}"
                if not SIMULATE and not target_dir.exists():
                    target_dir.mkdir(parents=True)

                # Handle duplicates
                # target_file = handle_duplicates(target_file, file_path)
                target_file, is_duplicate = handle_duplicates(target_file, file_path)
                if is_duplicate:
                    duplicate_target = DUPLICATE_DIR / file_path.name
                    if SIMULATE:
                        logging.info(f"Would move {file_path} to {duplicate_target}")
                    else:
                        logging.info(f"Moving {file_path} to {duplicate_target}")
                        try:
                            shutil.move(str(file_path), str(duplicate_target))
                        except Exception as e:
                            logging.error(f"Error moving {file_path} to {duplicate_target}: {e}")
                elif target_file:
                    if SIMULATE:
                        logging.info(f"Would move {file_path} to {target_file}")
                    else:
                        logging.info(f"Moving {file_path} to {target_file}")
                        try:
                            shutil.move(str(file_path), str(target_file))
                        except Exception as e:
                            logging.error(f"Error moving {file_path} to {target_file}: {e}")

# Process photos
process_images(PHOTOS_DIR, OUTPUT_DIR)

logging.info("Photos organized successfully!")
