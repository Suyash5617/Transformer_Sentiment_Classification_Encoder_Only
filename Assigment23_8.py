import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {

'Name': ['Amit', 'Sagar', 'Pooja'],
'Math' :[85, 90, 78],
'Science': [92, 88, 80],
'English': [75, 85, 82]

}

df = pd.DataFrame(data)
marks = df[df['Name'] == 'Amit'][['Math','Science','English']].values.flatten()

subject = ['Math','Science','English']
# print(marks)

plt.plot(subject,marks,marker = 'o')
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.title("Amit Report")
plt.grid(True)
plt.show()

