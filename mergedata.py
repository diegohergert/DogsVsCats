import os
import shutil

# --- CHANGE THESE ---
FOLDER_A = "dataset/animals/dog"
FOLDER_B = "dataset\kagglecatsanddogs_3367a\PetImages\Dog"
DEST_FOLDER = "all_dogs_merged"
# --------------------

def merge_folders_with_rename(source_folders, dest_dir):
    """
    Merges all files from a list of source folders into a single
    destination folder.
    
    If a file with the same name already exists in the destination,
    it renames the new file by appending '_1', '_2', etc.
    """
    
    # 1. Create the destination folder if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Destination folder set to: {dest_dir}")
    
    total_copied = 0
    total_renamed = 0
    
    # 2. Loop through each source folder
    for folder in source_folders:
        print(f"\nProcessing folder: {folder}")
        if not os.path.exists(folder):
            print(f"Warning: Source folder not found at {folder}. Skipping.")
            continue
            
        # 3. Loop through every file in the source folder
        for filename in os.listdir(folder):
            source_file = os.path.join(folder, filename)
            
            # Skip subdirectories, only copy files
            if not os.path.isfile(source_file):
                continue

            # 4. Set the initial destination path
            dest_file = os.path.join(dest_dir, filename)
            
            # 5. Check for name conflicts
            if not os.path.exists(dest_file):
                # No conflict: Just copy the file
                shutil.copy2(source_file, dest_file)
            else:
                # CONFLICT: Find a new name
                renamed = False
                base_name, extension = os.path.splitext(filename)
                i = 1
                
                while True:
                    # Create a new name, e.g., "image_1.jpg", "image_2.jpg"
                    new_filename = f"{base_name}_{i}{extension}"
                    new_dest_file = os.path.join(dest_dir, new_filename)
                    
                    if not os.path.exists(new_dest_file):
                        # Found a unique name!
                        shutil.copy2(source_file, new_dest_file)
                        print(f"  - Renamed '{filename}' to '{new_filename}'")
                        total_renamed += 1
                        renamed = True
                        break # Exit the while loop
                    
                    i += 1 # Try the next number
            
            total_copied += 1

    print(f"\n--- Merge Complete ---")
    print(f"Total files copied: {total_copied}")
    print(f"Total files renamed due to conflicts: {total_renamed}")

# --- Run the script ---
if __name__ == "__main__":
    source_list = [FOLDER_A, FOLDER_B]
    merge_folders_with_rename(source_list, DEST_FOLDER)