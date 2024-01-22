from icecream import ic
import time
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

filename = '/clusterfs/students/achmadjae/RA/02_deep3D/data/sho_train_40_15000.h5'
with h5py.File(filename, 'r') as hf:
    X = hf['feature'][:]
    y = hf['target'][:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# flat the X_train and X_test
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# regr = RandomForestRegressor(n_estimators=500, max_depth=80, random_state=42, criterion='absolute_error', n_jobs=-1, max_features=0.6, min_samples_leaf=10)
regr = XGBRegressor(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1)
# regr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
now = time.time()
regr.fit(X_train, y_train)
then = time.time()
ic(then-now)
y_pred = regr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
ic(mae)

