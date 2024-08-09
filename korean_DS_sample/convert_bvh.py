import bpy
import os
import glob
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
    bpy.context.scene.frame_end = frame_end
    bpy.ops.export_anim.bvh(filepath=export_filepath)

def modify_bvh_channels(bvh_path):
    with open(bvh_path, 'r') as file:
        lines = file.readlines()
    
    modified_lines = []
    first_occurrence = True
    
    for line in lines:
        if "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" in line:
            if first_occurrence:
                modified_lines.append(line)
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

    updated_motion_data = []
    for idx, line in enumerate(motion_data):
        values = line.split()
        filtered_values = [value for i, value in enumerate(values) if not (i % 6 in [0, 1, 2])]
        updated_motion_data.append(' '.join(filtered_values))

    with open(output_file, 'w') as file:
        file.writelines(lines[:motion_start_index])
        file.writelines('\n'.join(updated_motion_data) + '\n')

def convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path):
    clear()
    source_armature = import_bvh(source_bvh_path)
    target_armature = import_bvh(target_bvh_path)

    keyframes = get_keyframes([source_armature])
    frame_end = int(max(keyframes))
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

# Directory containing BVH files
directory = r'D:\motion-tokenizer\BEAT_dataset\beat_english_v0.2.1TEST'
target_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\bvhnormalized_output.bvh'
remap_path = os.path.abspath(r'D:\motion-tokenizer\korean_DS_sample\remap_preset.bmap')

# List all BVH files in the directory
bvh_files = glob.glob(os.path.join(directory, '**', '*.bvh'), recursive=True)
print(f"Found {len(bvh_files)} BVH files")

# Process each file
for i, bvh_file in enumerate(bvh_files, start=1):
    print(f"Processing file {i}/{len(bvh_files)}: {os.path.basename(bvh_file)}")
    output_bvh_path = bvh_file.replace(".bvh", "_converted.bvh")
    convert_bvh(bvh_file, target_bvh_path, output_bvh_path, remap_path)
    print(f"Completed: {os.path.basename(output_bvh_path)}")
