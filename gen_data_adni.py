import os
import subprocess
import numpy as np
import nibabel as nib

# Parse LUT file and build merged label mapping
def load_lut_and_merge_sides(lut_path):
    label_map = {}
    merged_map = {}
    current_index = 0

    with open(lut_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            label_id = int(parts[0])
            label_name = parts[1].lower()

            # Normalize left/right prefixes
            if label_name.startswith('left-'):
                norm_name = label_name.replace('left-', '')
            elif label_name.startswith('right-'):
                norm_name = label_name.replace('right-', '')
            elif label_name.startswith('ctx-lh-'):
                norm_name = label_name.replace('ctx-lh-', 'ctx-')
            elif label_name.startswith('ctx-rh-'):
                norm_name = label_name.replace('ctx-rh-', 'ctx-')
            else:
                norm_name = label_name

            if norm_name not in label_map:
                label_map[norm_name] = current_index
                current_index += 1
            merged_map[label_id] = label_map[norm_name]

    return merged_map

# Load LUT mapping once
lut_path = f"/tmp/mribin/freesurfer6_linux/FreeSurferColorLUT.txt"
merged_label_map = load_lut_and_merge_sides(lut_path)

for split in ["train"]:
    data_dir = f"/mnt/chrastil/users/marjanrsd/openbhb_fsoutput_adni/{split}"
    out_dir = f"/mnt/chrastil/users/marjanrsd/openbhb_fsoutput_adni/seg_labels"
    sub_lst = os.listdir(data_dir)
    sub_ids = []
    for d in sub_lst:
        if len(d) == 4:
            try:
                int(d)
                sub_ids.append(d)
            except:
                continue

    # @TODO remove! only for debug
    # sub_ids = sub_ids[:10]
    for sub_id in sub_ids:
        ##############
        # FOR LABELS #
        ##############
        sub_dir = os.path.join(data_dir, sub_id)
        input_mgz = os.path.join(sub_dir, "mri", "aparc+aseg.mgz")
        output_dir = os.path.join(out_dir, f"{split}")
        output_nii = os.path.join(output_dir, f"{sub_id}_aparc+aseg.nii.gz")

        # Check if input file exists
        if not os.path.exists(input_mgz):
            print(f"Missing file: {input_mgz}")
            continue

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        
        # Run mri_convert
        cmd = ["mri_convert", input_mgz, output_nii]
        try:
            subprocess.run(cmd)
            #subprocess.run(cmd, check=True)
            print(f"Converted {sub_id} → aparc+aseg.nii.gz")
        except subprocess.CalledProcessError:
            print(f"Labels conversion failed for {sub_id}")

        
        # Load the NIfTI file
        img = nib.load(output_nii)
        label_array = img.get_fdata().astype(np.int32) # Ensure integer for remapping
        label_array = label_array[48:-48, 48:-48, 48:-48]
        mapped_array = np.vectorize(lambda x: merged_label_map.get(x, 0))(label_array)
        original_shape = mapped_array.shape
        # Relabel to consecutive integers starting from 0
        _unique_labels, relabelled = np.unique(mapped_array, return_inverse=True)
        relabelled = relabelled.reshape(original_shape)

        if len(np.unique(relabelled)) != 61:
            print(f"missing ROIs. {len(np.unique(relabelled))} != 61 for {output_nii}")
            try:
                os.remove(output_nii)
            except FileNotFoundError:
                pass
            continue
        # Save as .npy
        npy_path = os.path.join(output_dir, f"{sub_id}_labels.npy")
        np.save(npy_path, relabelled)
        print(f"Saved {npy_path}")
        try:
            os.remove(output_nii)
        except FileNotFoundError:
            pass



        #################
        # FOR T1 VOXELS #
        #################
        input_mgz = os.path.join(sub_dir, "mri", "orig.mgz")
        output_nii = os.path.join(output_dir, f"{sub_id}_orig.nii.gz")

        # Check if input file exists
        if not os.path.exists(input_mgz):
            print(f"Missing file: {input_mgz}")
            continue

        
        # Run mri_convert
        cmd = ["mri_convert", input_mgz, output_nii]
        try:
            subprocess.run(cmd)
            #subprocess.run(cmd, check=True)
            print(f"Converted {sub_id} → orig.nii.gz")
        except subprocess.CalledProcessError:
            print(f"T1 conversion failed for {sub_id}")

        
        # Load the NIfTI file
        img = nib.load(output_nii)
        t1_array = img.get_fdata().astype(np.int32) # Ensure integer for remapping
        t1_array = t1_array[48:-48, 48:-48, 48:-48]

        # Save as .npy
        npy_path = os.path.join(output_dir, f"{sub_id}_T1.npy")
        np.save(npy_path, t1_array)
        print(f"Saved {npy_path}")
        try:
            os.remove(output_nii)
        except FileNotFoundError:
            pass
