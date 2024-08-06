import bpy
import os
import math

def gc():
    for i in range(10): bpy.ops.outliner.orphans_purge()

def clear():
    print("#############################  Clearing the scene...")
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.delete(use_global=False)
    gc()
    print("############################# Scene cleared.")

def import_bvh(filepath):
    print(f"############################# Importing BVH file from {filepath}...")
    bpy.ops.import_anim.bvh(filepath=filepath)
    armature = bpy.context.selected_objects[0]
    print(f"############################# Imported {armature.name}")
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
    print("############################# Starting retargeting process...")
    bpy.context.view_layer.objects.active = source_armature
    bpy.context.scene.source_rig = source_armature.name
    bpy.context.scene.target_rig = target_armature.name
    print("############################# Building bones list...")
    bpy.ops.arp.build_bones_list()
    print(f"############################# Importing remap configuration from {remap_path}...")
    bpy.ops.arp.import_config(filepath=remap_path)
    print("############################# Auto scaling...")
    bpy.ops.arp.auto_scale()
    print("############################# Retargeting animation...")
    bpy.ops.arp.retarget(frame_end=frame_end)
    print("############################# Retargeting complete.")

def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def scale_to_match(source, target):
    source_dimensions = source.dimensions
    target_dimensions = target.dimensions
    scale_factors = [s / t for s, t in zip(source_dimensions, target_dimensions)]
    average_scale_factor = sum(scale_factors) / len(scale_factors)
    target.scale *= average_scale_factor

def export_bvh(target_armature, export_filepath, frame_end):
    print(f"############################# Exporting retargeted animation to {export_filepath}...")
    bpy.ops.object.select_all(action='DESELECT')
    target_armature.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    bpy.context.scene.frame_end = frame_end  # Set the end frame for export
    bpy.ops.export_anim.bvh(filepath=export_filepath)
    print("############################# Export complete.")


def convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path):
    clear()  # Clear the blender scene
    
  
    source_armature = import_bvh(source_bvh_path)
    target_armature = import_bvh(target_bvh_path)

    # Get the number of frames and frame rate from the source armature
    keyframes = get_keyframes([source_armature])
    frame_end = int(max(keyframes))
    frame_rate = get_frame_rate(source_bvh_path)

    
    bpy.context.scene.render.fps = frame_rate

    
    scale_to_match(source_armature, target_armature)
    
    
    retarget(source_armature, target_armature, remap_path, frame_end)
    
    # Apply transforms to target armature to fix any rotation or scale issues
    apply_transforms(target_armature)
    
    
    export_bvh(target_armature, output_bvh_path, frame_end)

# Paths
source_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\4_lawrence_0_6_6.bvh'
target_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\bvhnormalized_output.bvh'
output_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\output_test.bvh'
remap_path = os.path.abspath(r'D:\motion-tokenizer\korean_DS_sample\remap_preset.bmap')

print("############################# Source BVH Path:", source_bvh_path)
print("############################# Target BVH Path:", target_bvh_path)
print("############################# Output BVH Path:", output_bvh_path)
print("############################# Remap Path:", remap_path)

# Execute the conversion process
convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path)
