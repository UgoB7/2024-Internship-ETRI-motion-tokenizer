import os

def delete_unwanted_files(directory):
    # Parcourir tous les fichiers dans le répertoire donné
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # Vérifier si le fichier ne se termine pas par les extensions désirées
            if not (file_name.endswith("_converted.bvh") or file_name.endswith("_translated.bvh")):
                file_path = os.path.join(root, file_name)
                try:
                    # Supprimer le fichier
                    os.remove(file_path)
                    print(f"Fichier supprimé : {file_path}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file_path}: {e}")

# Exemple d'utilisation
directory = "D:/motion-tokenizer/BEAT_dataset/Beat_dataset"
delete_unwanted_files(directory)
