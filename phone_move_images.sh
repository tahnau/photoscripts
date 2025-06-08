#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/media/my_images/ios_photobackup_folder/"
DEST_DIR="/media/my_images/000_INPUT"

# Find and copy files modified in the last 2 months
# find $SOURCE_DIR -type f -mtime -60 -exec cp {} $DEST_DIR \;
find "$SOURCE_DIR" -type f -mtime -60 -not -name '.photobackup' -exec mv {} $DEST_DIR \;

