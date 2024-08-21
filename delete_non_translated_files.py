import os

def delete_non_translated_files(directory):
    # Parcours des fichiers dans le répertoire
    for root, _, files in os.walk(directory):
        for file in files:
            # Vérifie si le fichier ne se termine pas par '_translated.bvh'
            if not file.endswith('_translated.bvh'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"################################################################### Error deleting {file_path}: {e}")

if __name__ == "__main__":
    directory = r"D:\motion-tokenizer\BEAT_dataset\Beat_dataset"
    delete_non_translated_files(directory)