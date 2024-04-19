import pandas as pd

file_path = 'celltocellholdout.csv'
data = pd.read_csv(file_path)

data_head = data.head()
data_info = data.info()
data_description = data.describe()

print(data_head, data_info, data_description)
