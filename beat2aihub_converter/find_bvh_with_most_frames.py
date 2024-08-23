import os

def count_frames_in_bvh(bvh_file):
    """Compte le nombre de frames dans un fichier BVH en lisant la section 'Frames:'."""
    with open(bvh_file, 'r') as file:
        for line in file:
            if line.strip().startswith("Frames:"):
                try:
                    # Prend en compte plusieurs espaces après "Frames:"
                    return int(line.split()[1])
                except (IndexError, ValueError):
                    return 0
    return 0

def find_bvh_with_most_frames(directory):
    """Trouve le fichier BVH avec le plus de frames dans un répertoire donné et ses sous-répertoires."""
    max_frames = 0
    bvh_with_max_frames = None

    for root, _, files in os.walk(directory):
        for file in files:
            # Vérifie que le fichier se termine par '_translated.bvh'
            if file.endswith('_translated.bvh'):
                bvh_path = os.path.join(root, file)
                num_frames = count_frames_in_bvh(bvh_path)
                #print(f"Frames: {num_frames}")
                if num_frames == 0:
                    print(f"##################################################### '{bvh_path}'  has 0 frames.")

                
                if num_frames > max_frames:
                    max_frames = num_frames
                    bvh_with_max_frames = bvh_path

    return bvh_with_max_frames, max_frames

def main():
    directories = [
        r"D:\motion-tokenizer\AIHUB_DATA",
        r"D:\motion-tokenizer\BEAT_dataset\Beat_dataset"
    ]

    for directory in directories:
        print(f"Searching in: {directory}")
        bvh_file, frame_count = find_bvh_with_most_frames(directory)
        
        if bvh_file:
            print(f"The BVH file with the most frames in '{directory}' is:")
            print(f"File: {bvh_file}")
            print(f"Number of frames: {frame_count}\n")
        else:
            print(f"No matching BVH files found in '{directory}'.\n")

if __name__ == "__main__":
    main()
