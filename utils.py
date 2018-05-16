import os
import shutil
import glob


def get_files(path: str, extension: str=""):
    files = glob.glob(os.path.join(os.getcwd(), path, extension))

    return files


def delete_create_dir(path: str):
    """Delete the directory and then recreates the same

    Keyword arguments:
    path -- directory to be deleted and then recreated
    """
    if os.path.exists(path):
        # delete the folder and its files
        shutil.rmtree(path)

    # recreate the folder
    os.makedirs(path)
