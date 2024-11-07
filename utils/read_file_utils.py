import os

def read_text_file(filename):
    """
    Function to read raw text file line-by-line and return list of sentence.
    """
    with open(filename, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    return lines


def get_folder_size(folder_path):
    total_size = 0
    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Get the size of each file in the folder
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Add file size (in bytes) to total size
            total_size += os.path.getsize(file_path)
    return total_size