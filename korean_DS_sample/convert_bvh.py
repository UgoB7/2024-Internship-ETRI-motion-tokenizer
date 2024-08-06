import bpy
import os
import math


'''def import_bvh(file_path, armature_name):
    bpy.ops.import_anim.bvh(filepath=file_path)
    imported_armature = bpy.context.selected_objects[0]
    imported_armature.name = armature_name
    return imported_armature

def remap_armatures(source_armature, target_armature, bmap_file):
    # Assurez-vous que Auto Rig Pro est activé
    if "auto_rig_pro" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="auto_rig_pro")

    # Sélectionnez les armatures source et cible
    bpy.context.view_layer.objects.active = bpy.data.objects[source_armature]
    source_obj = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.view_layer.objects.active = bpy.data.objects[target_armature]
    target_obj = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode='OBJECT')

    # Appliquez le remapping en utilisant le fichier .bmap
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    
    bpy.ops.auto_rig_pro.quick_rig(source=source_armature, target=target_armature, bmap_path=bmap_file)

    # Activer l'auto scale si nécessaire
    # Vous pouvez ajuster les paramètres de l'auto scale ici
    bpy.ops.auto_rig_pro.auto_scale()

# Remplacez les chemins des fichiers .bvh et le chemin du fichier .bmap
source_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\4_lawrence_0_7_7.bvh'
target_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\bvhnormalized_output.bvh'
bmap_file_path = os.path.abspath(r'D:\motion-tokenizer\korean_DS_sample\remap_preset.bmap')

source_armature_name = "Source_Armature"
target_armature_name = "Target_Armature"

# Importez les fichiers BVH
source_armature = import_bvh(source_bvh_path, source_armature_name)
target_armature = import_bvh(target_bvh_path, target_armature_name)

# Remappez les armatures
remap_armatures(source_armature.name, target_armature.name, bmap_file_path)
'''



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

def retarget(source_armature, target_armature, remap_path):
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
    keyframes = get_keyframes([source_armature])
    print("############################# Retargeting animation...")
    bpy.ops.arp.retarget(frame_end=int(max(keyframes)))
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

def export_bvh(target_armature, export_filepath):
    print(f"############################# Exporting retargeted animation to {export_filepath}...")
    bpy.ops.object.select_all(action='DESELECT')
    target_armature.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.export_anim.bvh(filepath=export_filepath)
    print("############################# Export complete.")

# Main function to handle importing, retargeting, rotating, and exporting the bvh files
def convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path):
    clear()  # Clear the blender scene
    
    # Import source and target BVH files
    source_armature = import_bvh(source_bvh_path)
    target_armature = import_bvh(target_bvh_path)

    # Scale the target armature to match the source armature
    scale_to_match(source_armature, target_armature)
    
    # Retarget animation from the source armature to the target armature using Auto Rig Pro remap configuration
    retarget(source_armature, target_armature, remap_path)
    
    # Apply transforms to target armature to fix any rotation or scale issues
    apply_transforms(target_armature)
    
    # Export the retargeted and rotated animation to a new BVH file
    export_bvh(target_armature, output_bvh_path)

# Paths
source_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\4_lawrence_0_7_7.bvh'
target_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\bvhnormalized_output.bvh'
output_bvh_path = r'D:\motion-tokenizer\korean_DS_sample\output_test.bvh'
remap_path = os.path.abspath(r'D:\motion-tokenizer\korean_DS_sample\remap_preset.bmap')

print("############################# Source BVH Path:", source_bvh_path)
print("############################# Target BVH Path:", target_bvh_path)
print("############################# Output BVH Path:", output_bvh_path)
print("############################# Remap Path:", remap_path)

# Execute the conversion process
convert_bvh(source_bvh_path, target_bvh_path, output_bvh_path, remap_path)