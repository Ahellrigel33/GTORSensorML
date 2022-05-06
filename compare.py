from SensorData import SensorData
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

S = SensorData(bin_predict=300, bin_average_window=50)
S.import_csv_files()
S.preprocess_data()
S.split_data()
S.aggregate_data(use_daata_files=False)

print(S.x_train.shape)
print(np.sum(np.isnan(S.x_train)))
print(S.x_train.dtype)


names = dict()
names['none'] = 'No'

reg = dict()
reg['ridge'] = Ridge(alpha=1); names['ridge'] = 'Ridge'
reg['lasso'] = Lasso(alpha=1); names['lasso'] = 'LASSO'
reg['krr']   = KernelRidge(kernel='rbf', gamma=1.0e-6, alpha=1e-5); names['krr'] = 'Kernel Ridge'
reg['svr']   = SVR(kernel='rbf', epsilon=100, gamma=1, C=1); names['svr'] = 'Support Vector'
reg['mlp']   = MLPRegressor(max_iter=1000, hidden_layer_sizes=(100,100), 
                            activation='relu', alpha=1); names['mlp'] = 'MLP'

scale = dict()
scale['stand'] = StandardScaler(); names['stand'] = 'Standard'

reduc = dict()
reduc['trsvd'] = TruncatedSVD(n_components=20); names['trsvd'] = 'Truncated SVD'
reduc['pca']   = PCA(n_components=100); names['pca'] = 'PCA'
reduc['kpca']  = KernelPCA(n_components=40, kernel="poly"); names['kpca'] = 'Kernel PCA'

tests = [
    ['ridge', 'none' , 'none'],
    ['lasso', 'none', 'none'],
    ['ridge', 'stand', 'none'],
    # ['ridge', 'stand', 'kpca'],
    # ['ridge', 'stand', 'none'],
    # ['ridge', 'stand', 'none'],
    # ['svr'  , 'stand', 'none'],
    # ['mlp'  , 'stand', 'none'],
    ]

# plt.figure()
# plt.plot(S.x_val[:,-8])
# plt.plot(S.x_val[:,-7])
# plt.figure()
# plt.plot(S.y_val[:,:])
# plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False)

print(axes)

for index, test in enumerate(tests) :
    x_train = S.x_train
    y_train = S.y_train
    x_val = S.x_val
    y_val = S.y_val

    title = names[test[0]] + ' Regression with ' + names[test[1]] + ' Scaling and ' + names[test[2]] + ' Dimensionality Reduction'
    print('\n====' + title + "====")

    # SCALE THE DATA (ZERO MEAN AND UNIT VARIANCE)
    if (test[1] != 'none') :
        scale[test[1]].fit(x_train)
        x_train = scale[test[1]].transform(x_train)
        x_val = scale[test[1]].transform(x_val)
    
    # DIMENSIONALITY REDUCTION
    if (test[2] != 'none') :
        reduc[test[2]].fit(x_train)
        x_train = reduc[test[2]].transform(x_train)
        x_val = reduc[test[2]].transform(x_val)
    
    # REGRESSION
    reg[test[0]].fit(x_train, y_train)
    y_train_pred = reg[test[0]].predict(x_train)
    y_val_pred = reg[test[0]].predict(x_val)

    axis = axes[index // 2][index % 2]
    axis.set_title(title)
    axis.plot(y_val, label=['y_val engine', 'y_val secondary'])
    axis.plot(y_val_pred, label=['y_val_pred engine', 'y_val_pred secondary'])
    axis.plot(x_val[:,-8], label='x_test[speed_engine_rpm]')
    axis.plot(x_val[:,-7], label='x_test[speed_secondary_rpm]')
    axis.set_ylim([0,4500])
    # axis.plot(y_val_pred, label='val_pred')
    axis.legend()

    print('Training   R^2: E=%.4f, S=%.4f' % (r2_score(y_train[:,0], y_train_pred[:,0]), r2_score(y_train[:,1], y_train_pred[:,1])))
    print('Validation R^2: E=%.4f, S=%.4f' % (r2_score(y_val[:,0], y_val_pred[:,0]), r2_score(y_val[:,1], y_val_pred[:,1])))
    print('Baseline   R^2: E=%.4f, S=%.4f' % (r2_score(y_val[:,0], S.x_val[:,-8]), r2_score(y_val[:,1], S.x_val[:,-7])))
    print('Training   MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_train[:,0], y_train_pred[:,0]), mean_squared_error(y_train[:,1], y_train_pred[:,1])))
    print('Validation MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_val[:,0], y_val_pred[:,0]), mean_squared_error(y_val[:,1], y_val_pred[:,1])))
    print('Baseline   MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_val[:,0], S.x_val[:,-8]), mean_squared_error(y_val[:,1], S.x_val[:,-7])))


plt.show()