import os
import shutil

# --- Modify these two paths (it is recommended to use absolute paths) ---
source_root = "/home/user/ETP-R1/extra_files"  # resource (extra_files)
target_root = "/home/user/ETP-R1"     # working directory
# habitat_source = "/home/user/habitat-lab-0.1.7/data/scene_datasets" # path to habitat scene_datasets, including mp3d subfolder
# ---------------------------------------

IGNORE_DIRS = {"scene_datasets"} 
# ---------------------------------------

def smart_copy_merge(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        if item in IGNORE_DIRS:
            print(f"[Ignore] Skipping excluded directory in copy: {item}")
            continue

        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            smart_copy_merge(src_path, dst_path)
        else:
            try:
                shutil.copy2(src_path, dst_path)
                print(f"[Copy] {src_path} -> {dst_path}")
            except Exception as e:
                print(f"[Error] Failed to copy {src_path}: {e}")

def force_create_symlink(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if os.path.exists(target):
        if os.path.islink(target):
            current_link = os.readlink(target)
            if current_link == source:
                print(f"[Skip] Habitat link is already correct: {target}")
                return
            else:
                print(f"[Update] Link target changed. Removing old link...")
                os.remove(target)
        
        elif os.path.isdir(target):
            print(f"[Conflict] Found directory at target: {target}")
            print(f"           Removing directory tree to make room for symlink...")
            try:
                shutil.rmtree(target)
            except Exception as e:
                print(f"[Error] Failed to remove directory {target}: {e}")
                return

    try:
        os.symlink(source, target)
        print(f"[External Link] Success: {target} -> {source}")
    except Exception as e:
        print(f"[Error] Failed to create symlink: {e}")

if __name__ == "__main__":
    print("--- Starting Deployment ---")
    
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    
    print(f"Copying extra_files from {source_root}...")
    smart_copy_merge(source_root, target_root)

    # print("-" * 30)

    # habitat_target = os.path.join(target_root, "data", "scene_datasets")
    
    # print(f"Linking Habitat dataset...")
    # force_create_symlink(habitat_source, habitat_target)

    print("--- Complete ---")