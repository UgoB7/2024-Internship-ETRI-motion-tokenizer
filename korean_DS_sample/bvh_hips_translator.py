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

    print(f"Initial ROOT Hips position: ({root_x}, {root_y}, {root_z})")
    
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

# Usage
input_bvh = 'korean_DS_sample/MM_D_C_FF_BB_S525S526_001_01.bvh'
output_bvh = 'korean_DS_sample/bvhnormalized_output.bvh'
normalize_and_align_bvh(input_bvh, output_bvh)
