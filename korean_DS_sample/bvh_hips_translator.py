import os
import re

def read_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    hierarchy = []
    motion = []
    is_hierarchy = True
    
    for line in lines:
        if is_hierarchy:
            hierarchy.append(line)
            if "MOTION" in line:
                is_hierarchy = False
        else:
            motion.append(line)
    
    return hierarchy, motion

def write_bvh(file_path, hierarchy, motion):
    with open(file_path, 'w') as file:
        file.writelines(hierarchy)
        file.writelines(motion)

def normalize_and_align_bvh(file_path, output_path):
    hierarchy, motion = read_bvh(file_path)
    
    # Extract initial ROOT Hips position
    motion_data = motion[2:]  # Skip first 2 lines (Frames, Frame Time)
    initial_frame = motion_data[0].strip().split()
    
    root_x = float(initial_frame[0])
    root_y = float(initial_frame[1])
    root_z = float(initial_frame[2])

    # Check if the initial position is already (0, 0, 0)
    if root_x == 0.0 and root_y == 0.0 and root_z == 0.0:
        print("Initial position is already (0, 0, 0). Skipping normalization.")
        return  # Skip this file if already at (0, 0, 0)
    else:
        print(f"Initial position: ({root_x}, {root_y}, {root_z}), for file: {file_path}")
    
    normalized_motion = [motion[0], motion[1]]  # Keep Frames and Frame Time lines
    for line in motion_data:
        values = line.strip().split()
        x = float(values[0]) - root_x
        y = float(values[1]) - root_y
        z = float(values[2]) - root_z
        normalized_values = [x, y, z] + [float(v) for v in values[3:]]
        normalized_line = " ".join(f"{v: .6f}" for v in normalized_values) + "\n"
        normalized_motion.append(normalized_line)
    
    # Combine hierarchy and normalized motion
    normalized_bvh = hierarchy + normalized_motion
    
    write_bvh(output_path, hierarchy, normalized_motion)

def normalize_all_bvh_in_directory(input_directory):
    bvh_count = 0


    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.bvh'):
                input_file_path = os.path.join(root, file)
                
                
                # Créer le chemin de sortie avec "_translated" ajouté au nom du fichier
                base_name, ext = os.path.splitext(file)
                output_file_name = f"{base_name}_translated{ext}"
                output_file_path = os.path.join(root, output_file_name)

                # Vérifier si le fichier "_translated" existe déjà
                if os.path.exists(output_file_path):
                    print(f"File {output_file_name} already exists. Skipping.")
                    continue  # Skip this file if the translated version exists

                bvh_count += 1

                # Normaliser le fichier BVH
                normalize_and_align_bvh(input_file_path, output_file_path)
                print(f"Normalized file saved to: {output_file_path}")


    print(f"\nTotal BVH files processed: {bvh_count}")


# Usage
input_directory = r'D:\motion-tokenizer\BEAT_dataset\beat_english_v0.2.1TEST_TRANS'

normalize_all_bvh_in_directory(input_directory)