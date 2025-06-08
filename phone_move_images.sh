#!/usr/bin/env bash

# ./phone_move_images.sh /media/my_images/ios_photobackup_folder/ /media/my_images/000_INPUT

# Usage check
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SOURCE_DIR DEST_DIR [DAYS]"
  exit 1
fi

# Args (default to 60 days)
SOURCE_DIR=$1
DEST_DIR=$2
DAYS=${3:-60}

# Move files modified in the last $DAYS days (excluding .photobackup)
find "$SOURCE_DIR" -type f -mtime -"$DAYS" \
     ! -name '.photobackup' \
     -exec mv -t "$DEST_DIR" {} +

# Fix ownership & permissions on the source tree
chown -R :halkousers "$SOURCE_DIR"
chmod -R g+rw       "$SOURCE_DIR"
chmod    g+s        "$SOURCE_DIR"
