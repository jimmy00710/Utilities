import os
from tqdm import tqdm_notebook

'''
Search function search file recursively

Input Argument - 
base path - base path of directory in which search is to be done.
extension - extension of file which has to be search. 

Output-
return list of files from in that directory.


This method is the fastest way of recursive file search in python
'''
def get_files_generator(base_path,file_extension):
    for entry in os.scandir(base_path):
        if entry.is_file() and entry.name.endswith(file_extension): #In this line more file extension condition can be added. 
            yield entry.path
        elif entry.is_dir():
            yield from get_files_generator(entry.path)
        else:
            pass
        
        

def search(base_path,file_extension):
    files = [fil for fil in tqdm_notebook(get_files_generator(base_path,file_extension))]
    return files
