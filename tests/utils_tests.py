import utils
import os, shutil, glob
from unittest import TestCase

current_dir = os.getcwd()


class UtilsTests(TestCase):

    def test_get_files(self):
        files = utils.get_files("parquets", "*.parquet")

        if len(files) > 0:
            assert all(f.endswith('.parquet') for f in files)

        files = utils.get_files("models", "*.pkl")

        if len(files) > 0:
            assert all(f.endswith('.pkl') for f in files)

    def test_delete_create_dir(self):
        directory = os.path.join(current_dir, "test_folder")

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        assert os.path.exists(directory)

        utils.delete_create_dir(directory)

        assert os.path.exists(directory)

        shutil.rmtree(directory)
