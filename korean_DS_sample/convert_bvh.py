import bpy
import os

def gc():
    for i in range(10):
        bpy.ops.outliner.orphans_purge()

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
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

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

def modify_bvh_channels(bvh_path):
    # Function to modify channels in a BVH file
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


def modify_bvh_offsets(bvh_path):
    # Function to modify OFFSET lines in a BVH file
    with open(bvh_path, 'r') as file:
        lines = file.readlines()
    
    modified_lines = []
    for line in lines:
        if line.strip().startswith("OFFSET"):
            parts = line.split()
            if len(parts) == 4:
                indent = line[:line.index("OFFSET")]
                modified_line = f"{indent}OFFSET {parts[1]} {parts[3]} {parts[2]}\n"
                print(f"Modified OFFSET line: {modified_line.strip()}")  # Print modified OFFSET line
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)
    
    with open(bvh_path, 'w') as file:
        file.writelines(modified_lines)



def remove_columns_from_bvh(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Identify the start of the motion data
    motion_start_index = None
    for i, line in enumerate(lines):
        if line.startswith("MOTION"):
            motion_start_index = i + 3  # Motion data starts after 3 lines from "MOTION"
            break

    # Extract the motion data
    motion_data = lines[motion_start_index:]

    # Initialize list to track removed columns for the first line
    removed_columns_indices = []
    removed_columns_values = []

    # Function to determine if a column should be removed
    def should_remove_column(index):
        # Remove columns 7, 8, 9, 13, 14, 15, 19, 20, 21, etc.
        for base in range(6, index + 1, 6):
            if index == base or index == base + 1 or index == base + 2:
                return True
        return False

    # Process the motion data to remove specified columns
    updated_motion_data = []
    schema = []
    for idx, line in enumerate(motion_data):
        values = line.split()
        filtered_values = []
        for i, value in enumerate(values):
            if should_remove_column(i):
                schema.append("x")
                if idx == 0:  # Track removed columns only for the first line
                    removed_columns_indices.append(i)
                    removed_columns_values.append(value)
            else:
                schema.append("o")
                filtered_values.append(value)
        updated_motion_data.append(' '.join(filtered_values))
        if idx == 0:  # Print removed columns for the first line
            print("Number of columns (first line):", len(values))
            print("Removed columns indices (first line):\n", removed_columns_indices, "\n")
            print("Removed columns values (first line):\n", removed_columns_values, "\n")
            print("Schema (first line):\n", ' '.join(schema), "\n")

    # Write the updated data back to a new file
    with open(output_file, 'w') as file:
        file.writelines(lines[:motion_start_index])
        file.writelines('\n'.join(updated_motion_data) + '\n')

from itertools import permutations
def modify_bvh_rotations(bvh_path, output_bvh_path):
    # Lire le fichier BVH
    with open(bvh_path, 'r') as file:
        lines = file.readlines()

    # Combinaisons possibles des rotations
    rotation_orders = list(permutations(["Xrotation", "Yrotation", "Zrotation"]))

    results = []

    for order in rotation_orders:
        order_str = " ".join(order)
        modified_lines = []

        for line in lines:
            for original_order in rotation_orders:
                original_order_str = " ".join(original_order)
                if original_order_str in line:
                    line = line.replace(original_order_str, order_str)
            modified_lines.append(line)

        # Créer le nouveau chemin de fichier en fonction de l'ordre de rotation
        base_name = os.path.splitext(output_bvh_path)[0]
        new_bvh_path = f"{base_name}_{order_str.replace(' ', '')}.bvh"
        
        # Enregistrer le fichier modifié avec le nouveau nom
        with open(new_bvh_path, 'w') as file:
            file.writelines(modified_lines)

        # Ajouter le résultat au retour de la fonction
        results.append(new_bvh_path)
    
    return results



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

    # Modify the channels in the output BVH file
    modify_bvh_channels(output_bvh_path)
    
    # Modify OFFSET lines in the output BVH file
    modify_bvh_offsets(output_bvh_path)

    remove_columns_from_bvh(output_bvh_path, output_bvh_path)

    # Appliquer les combinaisons de rotations
    rotation_files = modify_bvh_rotations(output_bvh_path, output_bvh_path)
    return rotation_files

    ########### A CHNANGER LES ROTATIONS PUIS VISUALISER LE RESULTAT
    
    # Remove specified columns from the output BVH file
    

    

# Paths
source_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\4_lawrence_0_6_6.bvh'
target_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\bvhnormalized_output.bvh'
output_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\output.bvh'
remap_path = os.path.abspath(r'D:\motion-tokenizer\korean_DS_sample\remap_preset.bmap')

print("############################# Source BVH Path:", source_bvh_path)
print("############################# Target BVH Path:", target_bvh_path)
print("############################# Output BVH Path:", output_bvh_path)
print("############################# Remap Path:", remap_path)

# Execute the conversion process and obtain the list of rotation files
rotation_files = convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path)

# Print the generated files
for file in rotation_files:
    print(f"Generated file: {file}")

# blender --background --python convert_bvh.py
