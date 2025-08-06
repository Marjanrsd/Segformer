import os
import subprocess
import shutil
import gzip
from pathlib import Path

# the size of the initial adni nifti files that are used as inputs to FS
# is(256, 256, 166) - the orig.mgz in FS is (256, 256, 256)
# our target for the numpy file to be used as inputs to our
# segformer model is (160, 160, 160)

# Paths
source_dir = "/mnt/chrastil/users/marjanrsd/openbhb_ct/cleaned_ADNI/ADNI"
target_dir = "/mnt/chrastil/users/marjanrsd/openbhb_ct/cleaned_ADNI/adni_t1"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Counter for naming
counter = 1

# Loop through all files in subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".nii"):
            src_path = os.path.join(root, file)
            new_filename = f"{counter:04d}.nii.gz"
            dest_path = os.path.join(target_dir, new_filename)

            # Compress and copy to target directory
            with open(src_path, 'rb') as f_in:
                with gzip.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(f"Copied and compressed: {src_path} -> {dest_path}")
            counter += 1

print(f"Done. Total files copied: {counter - 1}")

assert False


# change the subject folder
parent_dir = "/mnt/chrastil/users/marjanrsd/openbhb_ct/cleaned_ADNI/ADNI"

# Count only directories (i.e., subjects)
subject_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
print(f"Number of subjects: {len(subject_dirs)}")

assert False

# Folder where all subject directories are stored
parent_dir = "/mnt/chrastil/users/marjanrsd/openbhb_ct/cleaned_ADNI/ADNI"

for name in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, name)
    
    # Skip if not a directory
    if not os.path.isdir(full_path):
        continue

    # Expect format like "137_S_6883"
    if "_" in name:
        parts = name.split("_")
        if parts[-1].isdigit():
            new_name = parts[-1]  # e.g., "6883"
            new_path = os.path.join(parent_dir, new_name)
            
            # Avoid overwriting anything accidentally
            if not os.path.exists(new_path):
                print(f"Renaming: {name} ➜ {new_name}")
                os.rename(full_path, new_path)
            else:
                print(f"Skipping {name}: target {new_name} already exists.")


assert False





RAW_DATA_DIR = "/mnt/chrastil/users/marjanrsd/openbhb_ct/adni_unzipped"     
CLEAN_DATA_DIR = "/mnt/chrastil/users/marjanrsd/openbhb_ct/cleaned_ADNI"     

# Priority of folder names for T1 scans (highest first)
PRIORITY_FOLDERS = [
    "MPR__GradWarp__B1_Correction__N3",
    "MT1__GradWarp__N3m",
    "Accelerated_Sagittal_MPRAGE"
]

# dcm2niix command (make sure it's installed and in PATH)
DCM2NIIX_CMD = "dcm2niix"

def find_priority_folder(session_path):
    """Return the path of the highest priority folder that exists inside session_path, or None."""
    for folder_name in PRIORITY_FOLDERS:
        candidate = os.path.join(session_path, folder_name)
        if os.path.isdir(candidate):
            return candidate
    return None

def find_nifti_files(folder):
    """Recursively find all NIfTI files (.nii or .nii.gz) inside a folder."""
    nifti_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, f))
    return nifti_files

def convert_dicom_to_nifti(dicom_folder, output_folder):
    """Run dcm2niix to convert DICOMs in dicom_folder to NIfTI files in output_folder."""
    cmd = [DCM2NIIX_CMD, "-z", "y", "-f", "%p_%s", "-o", output_folder, dicom_folder]
    print(f"    Running dcm2niix: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def process_subject_session(subject_dir, session_dir):
    subject = os.path.basename(subject_dir)
    session = os.path.basename(session_dir)

    print(f"Processing Subject: {subject} Session: {session}")


    # Find best folder
    best_folder = find_priority_folder(session_dir)
    if best_folder is None:
        print(f"  No priority folders found in {session_dir}, skipping.")
        return

    print(f"  Selected folder: {best_folder}")

    # Prepare output folder
    output_folder = os.path.join(CLEAN_DATA_DIR, subject, session)
    os.makedirs(output_folder, exist_ok=True)

    # Check for NIfTI files in best_folder
    nifti_files = find_nifti_files(best_folder)

    if nifti_files:
        print(f"  Found {len(nifti_files)} NIfTI files. Copying...")
        for nifti in nifti_files:
            dst = os.path.join(output_folder, os.path.basename(nifti))
            shutil.copy2(nifti, dst)
    else:
        # No NIfTI found - convert DICOMs
        print(f"  No NIfTI files found, converting DICOMs in {best_folder}...")
        convert_dicom_to_nifti(best_folder, output_folder)



def main():
    # Walk through raw data dir
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        # Check if current dir is a session dir by testing if it contains any priority folders
        if any(folder in dirs for folder in PRIORITY_FOLDERS):
            session_dir = root
            subject_dir = os.path.dirname(session_dir)
            process_subject_session(subject_dir, session_dir)

if __name__ == "__main__":
    main()

