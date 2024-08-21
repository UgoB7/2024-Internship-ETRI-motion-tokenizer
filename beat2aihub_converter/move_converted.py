import os
import shutil
import time

# Répertoires source et destination
source_directory = r'D:\motion-tokenizer\BEAT_dataset\beat_english_v0.2.1'
destination_directory = r'D:\motion-tokenizer\BEAT_dataset_converted_sample\beat_english_v0.2.1'

# Obtenez l'heure actuelle
current_time = time.time()

# Initialisation du compteur de fichiers copiés
files_copied_count = 0

# Parcours tous les fichiers du répertoire source
for root, dirs, files in os.walk(source_directory):
    for file in files:
        # Vérifie si le fichier se termine par '_converted.bvh'
        if file.endswith('_converted.bvh'):
            # Chemin complet du fichier source
            file_path = os.path.join(root, file)
            
            # Vérifie si le fichier a été modifié il y a plus de 10 minutes
            file_mod_time = os.path.getmtime(file_path)
            if current_time - file_mod_time > 600:  # 600 secondes = 10 minutes
                
                # Recrée la structure des sous-répertoires dans le répertoire de destination
                relative_path = os.path.relpath(root, source_directory)
                destination_path = os.path.join(destination_directory, relative_path)
                
                # Crée le répertoire de destination s'il n'existe pas
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                
                # Copie le fichier dans le répertoire de destination
                shutil.copy(file_path, destination_path)
                files_copied_count += 1  # Incrémente le compteur
                print(f"File {file} copied to {destination_path}")

# Affiche le nombre total de fichiers copiés
print(f"\nTotal files copied: {files_copied_count}")
