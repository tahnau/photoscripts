#!/usr/bin/env python3

"""

python3 find_duplicates.py /path/to/scan # reports duplicates
python3 find_duplicates.py /path/to/scan --remove # removes duplicates


"""

import sys
import os
import hashlib
import argparse

def md5_hash(file_path, chunk_size=8192):
    """Compute the MD5 hash of a file in a memory-efficient manner."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(dirs):
    # Dictionary to group files by size: {size: [file_paths]}
    size_dict = {}

    # Collect all files under given directories
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path):
                    fsize = os.path.getsize(full_path)
                    size_dict.setdefault(fsize, []).append(full_path)

    # Now we have grouped all files by size.
    # Next step: For each size group with multiple files, compute hashes.
    duplicates = []  # List of (file_hash, [file_paths]) groups where duplicates found

    for size, files in size_dict.items():
        if len(files) > 1:
            # Potential duplicates, check md5 hash
            hash_dict = {}
            for file_path in files:
                file_hash = md5_hash(file_path)
                hash_dict.setdefault(file_hash, []).append(file_path)

            # Among these hashed groups, if any group has more than one file, they are duplicates.
            for file_hash, dupe_group in hash_dict.items():
                if len(dupe_group) > 1:
                    duplicates.append((file_hash, dupe_group))

    return duplicates

def main():
    parser = argparse.ArgumentParser(description="Find and optionally remove duplicate files.")
    parser.add_argument("dirs", nargs="+", help="Directory paths to search.")
    parser.add_argument("--remove", action="store_true", help="Remove duplicate files, keeping only the original.")
    parser.add_argument("--simulate", action="store_true", help="Simulate removal without actually removing files (only valid with --remove).")

    args = parser.parse_args()
    dirs = args.dirs
    remove_duplicates = args.remove
    simulate = args.simulate

    duplicates = find_duplicates(dirs)

    if duplicates:
        print("Duplicate files found:")
        for file_hash, group in duplicates:
            # Sort to find the original: by shortest filename length, then by oldest modification time
            group_sorted = sorted(group, key=lambda f: (len(os.path.basename(f)), os.path.getmtime(f)))
            original = group_sorted[0]
            print(f"{original} (original) {file_hash}")
            # The rest are duplicates
            for dup in group_sorted[1:]:
                print(f"{dup} (duplicate) {file_hash}")
                if remove_duplicates:
                    if simulate:
                        print(f"  Would remove: {dup}")
                    else:
                        try:
                            os.remove(dup)
                            print(f"  Removed: {dup}")
                        except OSError as e:
                            print(f"  Error removing {dup}: {e}")
    else:
        print("No duplicate files found.")

if __name__ == "__main__":
    main()
