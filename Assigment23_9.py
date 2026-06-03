import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {

 'Name': ['Amit', 'Sagar', 'Pooja'],

 'Math': [np.nan, 76, 88],

 'Science': [91, np nan, 85]
}

df = pd.DataFrame(data)

df.filter(df.mean())


