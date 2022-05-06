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

S = SensorData(bin_predict=100, bin_average_window=100, holdpoints=10)
S.import_csv_files()
S.preprocess_data()
S.split_data()
S.aggregate_data(use_daata_files=True)
# S.noise_injection(injection_probability=0.01, use_drouput=True)

print(S.x_train.shape)
print(np.sum(np.isnan(S.x_train)))
print(S.x_train.dtype)


names = dict()
names['none'] = 'No'

reg = dict()
reg['ridge'] = Ridge(alpha=1e2); names['ridge'] = 'Ridge'
reg['ridge2'] = Ridge(alpha=1); names['ridge2'] = 'Ridge'
reg['lasso'] = Lasso(alpha=1e0); names['lasso'] = 'LASSO'
reg['krr']   = KernelRidge(kernel='rbf', gamma=1.0e-2, alpha=1); names['krr'] = 'Kernel Ridge'
reg['svr']   = SVR(kernel='rbf', epsilon=100, gamma=1, C=1); names['svr'] = 'Support Vector'
reg['mlp']   = MLPRegressor(max_iter=1000, hidden_layer_sizes=(100,100), 
                            activation='relu', alpha=1); names['mlp'] = 'MLP'

scale = dict()
scale['stand'] = StandardScaler(with_mean=False); names['stand'] = 'Standard'

reduc = dict()
reduc['trsvd'] = TruncatedSVD(n_components=10); names['trsvd'] = 'Truncated SVD'
reduc['pca']   = PCA(n_components=5); names['pca'] = 'PCA'
reduc['kpca']  = KernelPCA(n_components=40, kernel="poly"); names['kpca'] = 'Kernel PCA'

tests = [
    ['ridge', 'none' , 'none'],
    ['ridge2', 'stand' , 'none'],
    # ['ridge', 'none' , 'kpca'],
    # ['lasso', 'none', 'none'],
    # ['lasso', 'stand', 'none'],
    # ['krr', 'stand', 'trsvd'],
    # ['ridge', 'stand', 'none'],
    # ['ridge', 'stand', 'kpca'],
    # ['ridge', 'stand', 'none'],
    # ['ridge', 'stand', 'none'],
    # ['svr'  , 'stand', 'none'],
    # ['svr'  , 'stand', 'none'],
    ]

# plt.figure()
# plt.plot(S.x_val[:,-8])
# plt.plot(S.x_val[:,-7])
# plt.figure()
# plt.plot(S.y_val[:,:])
# plt.show()

# plt.figure(figsize=(width,height))
fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(19.2,10.8), dpi=100)

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
    axis.plot(S.x_val[:,-5], label='x_test[speed_engine_rpm]')
    axis.plot(S.x_val[:,-4], label='x_test[speed_secondary_rpm]')
    axis.set_ylim([0,4500])
    # axis.plot(y_val_pred, label='val_pred')
    axis.legend()

    if ((test[0] == 'ridge' or test[0] == 'ridge2' or test[0] == 'lasso') and test[2] == 'none'):
        print("ENGINE WEIGHTS")
        S.print_weights(reg[test[0]].coef_[0,:])
        print("SECONDARY WEIGHTS")
        S.print_weights(reg[test[0]].coef_[1,:])

    print('Training   R^2: E=%.4f, S=%.4f' % (r2_score(y_train[:,0], y_train_pred[:,0]), r2_score(y_train[:,1], y_train_pred[:,1])))
    print('Validation R^2: E=%.4f, S=%.4f' % (r2_score(y_val[:,0], y_val_pred[:,0]), r2_score(y_val[:,1], y_val_pred[:,1])))
    print('Baseline   R^2: E=%.4f, S=%.4f' % (r2_score(y_val[:,0], S.x_val[:,-5]), r2_score(y_val[:,1], S.x_val[:,-4])))
    print('Training   MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_train[:,0], y_train_pred[:,0]), mean_squared_error(y_train[:,1], y_train_pred[:,1])))
    print('Validation MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_val[:,0], y_val_pred[:,0]), mean_squared_error(y_val[:,1], y_val_pred[:,1])))
    print('Baseline   MSE: E=%.4f, S=%.4f' % (mean_squared_error(y_val[:,0], S.x_val[:,-5]), mean_squared_error(y_val[:,1], S.x_val[:,-4])))


plt.show()