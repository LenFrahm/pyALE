import pickle
import numpy as np
import unittest
from utils.input_management import read_exp_excel, create_exp_df, create_task_df

filename = '/Users/lfrahm/Code/ALE/pyALE_devel/test/exp_info_test.xlsx'
lines, first_line_idxs, num_exp = read_exp_excel(filename)
exp_df = create_exp_df(lines, first_line_idxs, num_exp)
task_df = create_task_df(exp_df)

with open(f'test/read_exp_excel_test.pickle', 'rb') as f:
    exp_read, task_read = pickle.load(f)


class TestReadExpInfo(unittest.TestCase):
    def test_task_df(self):
        self.assertTrue(np.equal(task_read.values, task_df.values).all() == True)

        
    def test_exp_df(self):
        self.assertTrue(np.array_equal(exp_read.values[0][10], exp_df.values[0][10]))
        
if __name__ == '__main__':
    unittest.main()