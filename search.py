import os
from tqdm import tqdm_notebook


def get_files_generator(base_path,file_extension):
    for entry in os.scandir(base_path):
        if entry.is_file() and entry.name.endswith(file_extension):
            yield entry.path
        elif entry.is_dir():
            yield from get_files_generator(entry.path)
        else:
            pass
        
        

def search(base_path,file_extension):
    files = [fil for fil in tqdm_notebook(get_files_generator(base_path,file_extension))]
    return files
