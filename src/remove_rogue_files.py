import os
from typing import List

def list_dir(dir_path: str, file_to_remove: str) -> List[str]:
    dir_list = os.listdir(dir_path)
    if file_to_remove in dir_list:
        dir_list.remove(file_to_remove)
        
    return dir_list