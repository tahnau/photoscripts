#!/usr/bin/env python3
import sys
import os
import hashlib
import argparse

"""

python3 find_duplicates.py /path/to/scan # reports duplicates
python3 find_duplicates.py /path/to/scan --remove # removes duplicates

The script selects the original file to keep as the one with the shortest full path length and oldest modification time; duplicates are compared against this original.

"""

def md5_hash(file_path, chunk_size=8192):
    """Compute the MD5 hash of a file in a memory-efficient manner."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates(dirs):
    # Group files by size
    size_dict = {}
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path):
                    fsize = os.path.getsize(full_path)
                    size_dict.setdefault(fsize, []).append(full_path)

    duplicates = []
    for size, files in size_dict.items():
        if len(files) > 1:
            hash_dict = {}
            for file_path in files:
                file_hash = md5_hash(file_path)
                hash_dict.setdefault(file_hash, []).append(file_path)

            for file_hash, group in hash_dict.items():
                if len(group) > 1:
                    duplicates.append((file_hash, group))
    return duplicates


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find duplicate files. By default, only lists duplicates and shows which files would be removed. "
            "Use --remove to actually delete duplicates. "
            "The script selects the original file to keep as the one with the shortest full path length and oldest modification time; duplicates are compared against this original."
        )
    )
    parser.add_argument(
        "dirs", nargs="+", help="Directory paths to search for duplicate files."
    )
    parser.add_argument(
        "--remove", action="store_true",
        help=(
            "When specified, remove duplicate files, keeping only the original. "
            "Without this flag, the script only lists duplicates and indicates which files would be removed. "
            "Original selection uses shortest full path length and oldest modification time criteria."
        )
    )

    args = parser.parse_args()
    dirs = args.dirs
    remove_duplicates = args.remove

    duplicates = find_duplicates(dirs)

    if duplicates:
        print("Duplicate files found:")
        for file_hash, group in duplicates:
            # Determine original by shortest full path length then oldest mod time
            sorted_group = sorted(
                group,
                key=lambda f: (len(f), os.path.getmtime(f))
            )
            original = sorted_group[0]
            print(f"{original} (original) {file_hash}")
            for dup in sorted_group[1:]:
                print(f"{dup} (duplicate) {file_hash}")

                if remove_duplicates:
                    try:
                        os.remove(dup)
                        print(f"  Removed: {dup}")
                    except OSError as e:
                        print(f"  Error removing {dup}: {e}")
                else:
                    print(f"  Would remove: {dup}")
    else:
        print("No duplicate files found.")

if __name__ == "__main__":
    main()
