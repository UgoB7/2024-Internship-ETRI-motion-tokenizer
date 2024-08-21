import bpy
import os
from math import radians

def gc():
    for i in range(10):
        bpy.ops.outliner.orphans_purge()

def clear():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.delete(use_global=False)
    gc()

def import_bvh(filepath):
    bpy.ops.import_anim.bvh(filepath=filepath)
    armature = bpy.context.selected_objects[0]
    return armature

def get_keyframes(obj_list):
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append(int(x))
    return keyframes

def get_frame_rate(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            if "Frame Time:" in line:
                frame_time = float(line.split(":")[1].strip())
                return int(1 / frame_time)
    return 24  # Default frame rate if not found

def retarget(source_armature, target_armature, remap_path, frame_end):
    bpy.context.view_layer.objects.active = source_armature
    bpy.context.scene.source_rig = source_armature.name
    bpy.context.scene.target_rig = target_armature.name
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=remap_path)
    bpy.ops.arp.auto_scale()
    bpy.ops.arp.retarget(frame_end=frame_end)

def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

def scale_to_match(source, target):
    source_dimensions = source.dimensions
    target_dimensions = target.dimensions
    scale_factors = [t / s for s, t in zip(source_dimensions, target_dimensions)]
    average_scale_factor = sum(scale_factors) / len(scale_factors)
    source.scale *= average_scale_factor

def export_bvh(target_armature, export_filepath, frame_end):
    bpy.ops.object.select_all(action='DESELECT')
    target_armature.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    bpy.context.scene.frame_end = frame_end  # Set the end frame for export
    bpy.ops.export_anim.bvh(filepath=export_filepath)

def modify_bvh_channels(bvh_path):
    with open(bvh_path, 'r') as file:
        lines = file.readlines()
    
    modified_lines = []
    first_occurrence = True
    
    for line in lines:
        if "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" in line:
            if first_occurrence:
                modified_lines.append(line)  # Keep the first occurrence unchanged
                first_occurrence = False
            else:
                modified_lines.append(line.replace("CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation", 
                                                   "CHANNELS 3 Zrotation Xrotation Yrotation"))
        else:
            modified_lines.append(line)
    
    with open(bvh_path, 'w') as file:
        file.writelines(modified_lines)


def remove_columns_from_bvh(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    motion_start_index = None
    for i, line in enumerate(lines):
        if line.startswith("MOTION"):
            motion_start_index = i + 3
            break

    motion_data = lines[motion_start_index:]
    removed_columns_indices = []
    removed_columns_values = []

    def should_remove_column(index):
        for base in range(6, index + 1, 6):
            if index == base or index == base + 1 or index == base + 2:
                return True
        return False

    updated_motion_data = []
    schema = []
    for line in motion_data:
        values = line.split()
        filtered_values = []
        for i, value in enumerate(values):
            if should_remove_column(i):
                schema.append("x")
                if len(updated_motion_data) == 0:  # Track removed columns only for the first line
                    removed_columns_indices.append(i)
                    removed_columns_values.append(value)
            else:
                schema.append("o")
                filtered_values.append(value)
        updated_motion_data.append(' '.join(filtered_values))

    with open(output_file, 'w') as file:
        file.writelines(lines[:motion_start_index])
        file.writelines('\n'.join(updated_motion_data) + '\n')



def convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path):
    clear()  # Clear the blender scene
    
    source_armature = import_bvh(source_bvh_path)
    target_armature = import_bvh(target_bvh_path)

    # keyframes = get_keyframes([source_armature])
    # frame_end = int(max(keyframes))
    # frame_rate = get_frame_rate(source_bvh_path)

    print('get_keyframes')
    # keyframes = get_keyframes([source_armature])
    # frame_end = int(max(keyframes))
    a = source_armature.animation_data.action
    frame_start, frame_end = map(int, a.frame_range)
    print(frame_end)
    print('get_frame_rate')
    frame_rate = get_frame_rate(source_bvh_path)

    bpy.context.scene.render.fps = frame_rate

    scale_to_match(source_armature, target_armature)
    
    retarget(source_armature, target_armature, remap_path, frame_end)
    
    apply_transforms(target_armature)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    target_armature.rotation_euler[0] -= radians(90)

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    export_bvh(target_armature, output_bvh_path, frame_end)

    modify_bvh_channels(output_bvh_path)

    remove_columns_from_bvh(output_bvh_path, output_bvh_path)

# Paths
directory_path = r'D:\motion-tokenizer\BEAT_dataset\beat_english_v0.2.1'
target_bvh_path = r'D:\motion-tokenizer\beat2aihub_converter\bvhnormalized_output.bvh'
remap_path = os.path.abspath(r'D:\motion-tokenizer\beat2aihub_converter\remap_preset.bmap')
start_subdir = 1
end_subdir = 10

# Left  to  do:
# 11-20  
# 21-30

dataset_path = directory_path
subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
filtered_subdirs = [d for d in subdirs if d.isdigit() and start_subdir <= int(d) <= end_subdir]


bvh_files = []
for subdir in filtered_subdirs:
    subdir_path = os.path.join(dataset_path, subdir)
    for root, dirs, files in os.walk(subdir_path):
        bvh_files.extend([os.path.join(root, file) for file in files if file.endswith('.bvh')])
print(f"Found {len(bvh_files)} .bvh files in the directory")


total_files = len(bvh_files)


# processing of the .bvh
for index, bvh_file in enumerate(bvh_files, start=1):
    bvh_subdir = os.path.dirname(bvh_file)
    bvh_filename = os.path.basename(bvh_file)

    # Check if the ..._converted file already exists
    if bvh_filename.endswith('_converted.bvh'):
        print(f"#############################################  Skipping ..._converted file: {bvh_filename}")
        continue
    
    output_bvh_filename = bvh_filename.replace('.bvh', '_converted.bvh')

    output_bvh_path = os.path.join(bvh_subdir, output_bvh_filename)

    # Check if the converted file already exists
    if os.path.exists(output_bvh_path):
        print(f"#############################################  Skipping already converted file: {bvh_filename}")
        continue

    print(f"Processing file {index}/{total_files}: {bvh_filename}")
    convert_bvh(bvh_file, target_bvh_path, output_bvh_path, remap_path)

    progress = int((index / total_files) * 50)
    print(f"[{'#' * progress}{'.' * (50 - progress)}] {index}/{total_files} files processed")

#  blender --background --python convert_bvh.py