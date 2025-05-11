#!/usr/bin/env python
import os
import glob
from pathlib import Path
import shutil

def find_and_delete_index_files():
    """Find and delete the corrupted PaperQA index files (files.zip)"""
    
    # Common index file locations
    home = os.path.expanduser("~")
    common_locations = [
        os.path.join(home, ".cache", "paperqa"),
        os.path.join(home, ".paperqa"),
        os.path.join(home, "Library", "Caches", "paperqa"),
    ]
    
    # Project-specific locations
    project_root = os.path.join(home, "Documents", "GitLab_Projects", "master_project", "xx823")
    project_locations = glob.glob(os.path.join(project_root, "**", "pqa_index"), recursive=True)
    
    # All locations to check
    all_locations = common_locations + project_locations
    
    # Find all files.zip files in these locations
    index_files = []
    for location in all_locations:
        if os.path.exists(location):
            print(f"Checking {location}...")
            for root, _, files in os.walk(location):
                if "files.zip" in files:
                    index_path = os.path.join(root, "files.zip")
                    index_files.append(index_path)
    
    # Delete found files
    if not index_files:
        print("No index files found!")
    else:
        print(f"\nFound {len(index_files)} index files:")
        for file_path in index_files:
            print(f"  {file_path}")
        
        confirm = input("\nDelete these files? (y/n): ")
        if confirm.lower() == 'y':
            for file_path in index_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print("\nAll index files deleted. Run your evaluation again.")
        else:
            print("Operation cancelled.")

if __name__ == "__main__":
    find_and_delete_index_files()