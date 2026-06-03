import pandas as pd
import numpy as np

data = {

'Name': ['Amit', 'Sagar', 'Pooja'],
'Math' :[85, 90, 78],
'Science': [92, 88, 80],
'English': [75, 85, 82]

}

df = pd.DataFrame(data)
print("Shape : ",df.shape)
print("Shape : ",df.columns)
print("Shape : ",df.dtypes)