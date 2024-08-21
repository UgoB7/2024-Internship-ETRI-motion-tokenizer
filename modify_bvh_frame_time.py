import os

def modify_bvh_frame_time(input_bvh_path, original_frame_time=0.008333, new_frame_time=0.033333, rename_output=False):
    with open(input_bvh_path, 'r') as file:
        lines = file.readlines()

    header = []
    motion_data = []
    in_motion_section = False
    motion_start_line = None
    motion_end_line = None

    # Parcourir le fichier pour séparer l'en-tête des données de mouvement
    for index, line in enumerate(lines):
        if line.startswith("Frame Time:"):
            # Remplacer le temps de frame
            line = f"Frame Time:    {new_frame_time:.6f}\n"
            header.append(line)
        elif line.startswith("Frames:"):
            # On va mettre à jour cette valeur plus tard, donc on la garde telle quelle pour l'instant
            header.append(line)
        elif line.strip() == "MOTION":
            in_motion_section = True
            motion_start_line = index + 1  # Ligne où commencent les données de motion
            header.append(line)
        elif in_motion_section:
            motion_data.append(line)
            motion_end_line = index  # On garde en mémoire la dernière ligne de motion
        else:
            header.append(line)

    if motion_start_line is not None and motion_end_line is not None:
        # Calculer le nombre réel de frames dans la section MOTION
        original_num_frames = (motion_end_line - motion_start_line) + 1

        # Calculer le facteur de sous-échantillonnage
        step = int(new_frame_time / original_frame_time)

        # Sous-échantillonnage des frames
        sampled_motion_data = motion_data[::step]

        # Mettre à jour le nombre de frames dans l'en-tête
        new_num_frames = len(sampled_motion_data)
        for i in range(len(header)):
            if header[i].startswith("Frames:"):
                header[i] = f"Frames: {new_num_frames}\n"
                break

        # Combiner l'en-tête et les nouvelles données de mouvement
        modified_bvh = header + sampled_motion_data

        # Définir le nom du fichier de sortie
        output_bvh_path = input_bvh_path
        if rename_output:
            base_name, ext = os.path.splitext(input_bvh_path)
            output_bvh_path = f"{base_name}_frame_modified{ext}"

        # Écrire le fichier BVH modifié
        with open(output_bvh_path, 'w') as file:
            file.writelines(modified_bvh)

        print(f"Fichier BVH modifié : {output_bvh_path}")

def modify_all_bvh_files_in_directory(input_directory, original_frame_time=0.008333, new_frame_time=0.033333, rename_output=False):
    # Parcourir tous les fichiers dans le répertoire d'entrée
    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            if file_name.endswith(".bvh"):
                input_bvh_path = os.path.join(root, file_name)
                
                # Modifier le fichier BVH
                modify_bvh_frame_time(input_bvh_path, original_frame_time, new_frame_time, rename_output)

# Exemple d'utilisation
input_directory = "D:/motion-tokenizer/BEAT_dataset/Beat_dataset"
rename_output = False  # Mettre à True pour renommer les fichiers

modify_all_bvh_files_in_directory(input_directory, rename_output=rename_output)
