import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

TRAINPOINTS = 1000
TESTRANGE = 15000, 20000
# TESTRANGE = 30000,35000
TESTPOINTS = TESTRANGE[1] - TESTRANGE[0]
CUTOFF = 20000
HOLDPOINTS = 10
PREDICT = 60
EPS = 0.05
AVG_WINDOW = 30
DELTA = 1

csv = np.genfromtxt('Cody_4LapTest1_BIN.csv', dtype=float, delimiter=",", names=True)
# print(csv.dtype.names)

speed_engine_rpm = np.convolve(np.array(csv['speed_engine_rpm']), np.ones(AVG_WINDOW)/AVG_WINDOW, 'same') 
speed_secondary_rpm = np.convolve(np.array(csv['speed_secondary_rpm']), np.ones(AVG_WINDOW)/AVG_WINDOW, 'same') 
imu_acceleration_x = np.convolve(np.array(csv['imu_acceleration_x']), np.ones(AVG_WINDOW)/AVG_WINDOW, 'same')
imu_acceleration_y = np.convolve(np.array(csv['imu_acceleration_y']), np.ones(AVG_WINDOW)/AVG_WINDOW, 'same') 
imu_acceleration_z = np.convolve(np.array(csv['imu_acceleration_z']), np.ones(AVG_WINDOW)/AVG_WINDOW, 'same') 

datanames = ['time', 'engine', 'secondary', 'pedal_lds', 'imux', 'imuy', 'imuz', 'frontbrake']
data = np.array([
    csv['time_auxdaq_us'], speed_engine_rpm, speed_secondary_rpm, csv['lds_pedal_mm'], 
    imu_acceleration_x, imu_acceleration_y, imu_acceleration_z, csv['pressure_frontbrake_psi']
]).T
# speed_engine_rpm = speed_engine_rpm[0:86000]
(DATAPOINTS, DATAFEATURES) = data.shape

x_train = np.empty((TRAINPOINTS, DATAFEATURES*HOLDPOINTS + 1))
y_train = np.empty((TRAINPOINTS,2))
x_test = np.empty((TESTPOINTS, DATAFEATURES*HOLDPOINTS + 1))
y_test = np.empty((TESTPOINTS,2))

rng = np.random.default_rng(0)
# print(int((DATAPOINTS - HOLDPOINTS - PREDICT)/PREDICT))
numbers = rng.choice((DATAPOINTS - HOLDPOINTS - PREDICT - CUTOFF - TESTRANGE[1]), size=TRAINPOINTS, replace=False)
for i, ibase in enumerate(numbers) :
    # base = np.random.randint(0,DATAPOINTS - HOLDPOINTS - PREDICT)
    base = ibase + TESTRANGE[1]
    for hp in range(HOLDPOINTS) :
        x_train[i,hp*DATAFEATURES:(hp+1)*DATAFEATURES] = data[base+hp,:]
    x_train[i,DATAFEATURES*HOLDPOINTS] = 1

    y_train[i,0] = speed_engine_rpm[base + HOLDPOINTS + PREDICT]
    y_train[i,1] = speed_secondary_rpm[base + HOLDPOINTS + PREDICT]

for i, base in enumerate(range(TESTRANGE[0], TESTRANGE[1])) :
    for hp in range(HOLDPOINTS) :
        x_test[i,hp*DATAFEATURES:(hp+1)*DATAFEATURES] = data[base+hp,:]
    x_test[i,DATAFEATURES*HOLDPOINTS] = 1

    y_test[i,0] = speed_engine_rpm[base + HOLDPOINTS + PREDICT]
    y_test[i,1] = speed_secondary_rpm[base + HOLDPOINTS + PREDICT]


scalex = StandardScaler()
scalex.fit(x_train)
x_scaletrain = scalex.transform(x_train)
x_scaletest = scalex.transform(x_test)

# pca = TruncatedSVD(n_components=20)
# pca.fit(x_train, y_train)
# x_pcatrain = pca.transform(x_train)
# x_pcatest = pca.transform(x_test)

# kpca = KernelPCA(n_components=100, kernel="poly")
# kpca.fit(x_scaletrain)
# x_kpcatrain = kpca.transform(x_scaletrain)
# x_kpcatest = kpca.transform(x_scaletest)

ls = Ridge(alpha=DELTA)
# ls = Lasso(alpha=DELTA)
# ls = KernelRidge(kernel='rbf', gamma=1.0e-6, alpha=1e-5)
# ls = SVR(kernel='rbf', epsilon=100, gamma=1, C=1)
# ls = MLPRegressor(max_iter=1000, hidden_layer_sizes=(100,100), activation='relu', alpha=1)
ls.fit(x_train, y_train)
y_trainout = ls.predict(x_train)
y_testout = ls.predict(x_test)
# ls.fit(x_scaletrain, y_train)
# y_trainout = ls.predict(x_scaletrain)
# y_testout = ls.predict(x_scaletest)
# ls.fit(x_kpcatrain, y_train)
# y_trainout = ls.predict(x_kpcatrain)
# y_testout = ls.predict(x_kpcatest)
# ls.fit(x_pcatrain, y_train)
# y_trainout = ls.predict(x_pcatrain)
# y_testout = ls.predict(x_pcatest)

# weights = ls.coef_
weights = []

# weights = np.linalg.inv(x_train.T @ x_train + DELTA*np.identity(x_train.shape[1])) @ x_train.T @ y_train
# y_trainout = x_train @ weights
# y_testout = x_test @ weights

# weights = np.linalg.inv(x_pcatrain.T @ x_pcatrain + LAM*np.identity(x_pcatrain.shape[1])) @ x_pcatrain.T @ y_train
# y_trainout = x_pcatrain @ weights
# y_testout = x_pcatest @ weights

for i, weight in enumerate(weights) :
    print("%s-%d: %.5f" % (datanames[i%DATAFEATURES], i/DATAFEATURES, weight))
    # if ((i-1)%DATAFEATURES == 0) :
    #     print("Engine Weight %d: %.4f" % ((i-1)/DATAFEATURES, weight))
    # if ((i-3)%DATAFEATURES == 0) :
    #     print("LDS Weight %d: %.4f" % ((i-3)/DATAFEATURES, weight))

ratio_test = y_test[:,0] / y_test[:,1]
ratio_testout = y_testout[:,0] / y_testout[:,1]
ratio_base = x_test[:,-(DATAFEATURES)] / x_test[:,-(DATAFEATURES-1)]

errortrain = 1 - ((y_train - y_trainout)**2).sum(axis=0) / ((y_train - y_train.mean(axis=0))**2).sum(axis=0)
errorbase = 1 - ((y_test - np.array([x_test[:,-(DATAFEATURES)], x_test[:,-(DATAFEATURES-1)]]).T)**2).sum(axis=0) / ((y_test - y_test.mean(axis=0))**2).sum(axis=0)
errorout = 1 - ((y_test - y_testout)**2).sum(axis=0) / ((y_test - y_test.mean(axis=0))**2).sum(axis=0)
errorratiobase = 1 - ((ratio_test - ratio_base)**2).sum(axis=0) / ((ratio_test - ratio_test.mean(axis=0))**2).sum(axis=0)
errorratioout = 1 - ((ratio_test - ratio_testout)**2).sum(axis=0) / ((ratio_test - ratio_test.mean(axis=0))**2).sum(axis=0)
print("Training Performance: ", errortrain)
print("Baseline Performance: ", errorbase)
print("Testing  Performance: ", errorout)
print("Base Rat Performance: ", errorratiobase)
print("Test Rat Performance: ", errorratioout)

fig, axes = plt.subplots(nrows=2,ncols=1)
axes[0].plot(ratio_test, label='ratio_test')
axes[0].plot(ratio_testout, label='ratio_testout')
axes[0].plot(ratio_base, label='ratio_base')
axes[0].legend()

axes[1].plot(y_test, label=['y_test engine', 'y_test secondary'])
axes[1].plot(y_testout, label=['y_testout engine', 'y_testout secondary'])
axes[1].plot(x_test[:,-(DATAFEATURES)], label='x_test[speed_engine_rpm]')
axes[1].plot(x_test[:,-(DATAFEATURES-1)], label='x_test[speed_secondary_rpm]')
axes[1].legend()

# plt.figure()
# plt.plot(y_test, label=['y_test eng', 'y_test sec'])
# plt.plot(y_testout, label='y_testout')
# plt.plot(x_test[:,-(DATAFEATURES)], label='x_test[speed_engine_rpm]')
# plt.plot(x_test[:,-(DATAFEATURES-1)], label='x_test[speed_secondary_rpm]')
# plt.legend()

plt.show()
