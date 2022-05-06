from SensorData import SensorData
import numpy as np
import matplotlib.pyplot as plt
S = SensorData()
S.import_csv_files()
S.preprocess_data()
S.split_data()
S.aggregate_data()

print(S.x_train.shape)
print(np.sum(np.isnan(S.x_train)))
print(S.x_train.dtype)

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

names = dict()

reg = dict()
reg['ridge'] = Ridge(alpha=1)
reg['lasso'] = Lasso(alpha=1)
reg['krr']   = KernelRidge(kernel='rbf', gamma=1.0e-6, alpha=1e-5)
reg['svr']   = SVR(kernel='rbf', epsilon=100, gamma=1, C=1)
reg['mlp']   = MLPRegressor(max_iter=1000, hidden_layer_sizes=(100,100), 
                            activation='relu', alpha=1)

scale = dict()
scale['stand'] = StandardScaler()

reduc = dict()
reduc['trsvd'] = TruncatedSVD(n_components=20)
reduc['pca']   = PCA(n_components=100)
reduc['kpca']  = KernelPCA(n_components=100, kernel="poly")

tests = [
    ['ridge', 'none' , 'none'],
    ['ridge', 'stand', 'none'],
    # ['lasso', 'stand', 'none'],
    ['svr'  , 'stand', 'none'],
    ['mlp'  , 'stand', 'none'],
    ]

fig, axes = plt.subplots(nrows=int(np.floor(np.sqrt(len(tests)))), 
                         ncols=int(np.ceil(np.sqrt(len(tests)))))


for index, test in enumerate(tests) :
    x_train = S.x_train
    y_train = S.y_train
    x_val = S.x_val
    y_val = S.y_val

    # SCALE THE DATA (ZERO MEAN AND UNIT VARIANCE)
    if (test[1] != 'none') :
        scale[test[1]].fit(x_train)
        x_train = scale[test[1]].transform(x_train)
        x_val = scale[test[1]].transform(x_val)
    
    # DIMENSIONALITY REDUCTION
    if (test[2] != 'none') :
        reduc[test[2]].fit(x_train)
        x_train = reduc[test[2]].transform(x_train)
    
    # REGRESSION
    reg[test[0]].fit(x_train, y_train)
    y_train_pred = test[0].predict(x_train)
    y_val_pred = test[0].predict(x_val)

    print("Training   R^2: ", r2_score(y_train, y_train_pred))
    print("Validation R^2: ", r2_score(y_train, y_train_pred))
    print("Training   MSE: ", mean_squared_error(y_train, y_train_pred))
    print("Validation MSE: ", mean_squared_error(y_train, y_train_pred))

    axis = axes[index % 2][index / 2]

