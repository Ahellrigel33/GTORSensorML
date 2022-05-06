import numpy as np
from SensorData import SensorData

S = SensorData()
S.import_csv_files()
S.preprocess_data()
S.split_data()
S.aggregate_data()

print(S.x_train.shape)
print(np.sum(np.isnan(S.x_train)))
print(S.x_train.dtype)
