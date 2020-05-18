import pandas as pd
from os import listdir
import os
f1  = listdir('test_data_1')
f2  = listdir('test_data_2')

for file1, file2 in zip(f1,f2):
    df1= pd.read_csv(os.path.join('test_data_1',file1))
    df2 = pd.read_csv(os.path.join('test_data_2',file2))
    print(df1)
    print(df2)