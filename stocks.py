import time
import math
import quandl
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']) * 100.0
df['PCT_CHANGE'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']) * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
print forecast_out

df['label'] = df[forecast_col].shift(-forecast_out)

print df.head()

X = np.array(df.drop(['label', 'Adj. Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

print len(X), len(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as file:
	pickle.dump(clf, file)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(datetime.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S").timetuple())

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day

	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print df.head()
print ""
print df.tail()
print ""

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

print "Forecast: {} \n Accuracy: {} Days: {}".format(forecast_set, accuracy*100, forecast_out)

plt.show()
