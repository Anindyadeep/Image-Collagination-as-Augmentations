import os
from run import get_test_results

test_path = str(os.getcwd())[:-4] + 'test_folder'
get_test_results(test_path=test_path)
