import pandas as pd
import numpy as np

data = {

'Name': ['Amit', 'Sagar', 'Pooja'],
'Math' :[85, 90, 78],
'Science': [92, 88, 80],
'English': [75, 85, 82]

}

df = pd.DataFrame(data)

print("Student who scored more than 85n in science ")

print(df[df['Science'] > 85])
