
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib import style
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm

style.use('ggplot')

df = quandl.get("EOD/HD", authtoken="KXs7ei6aAkAu5zhWUMsQ")

# calculate new features... you may also want to calculate other indicators that are important to you like SAR, MACD, EMA, interest rates, money supply, etc...
df['100ma'] = df['Adj_Close'].rolling(window=100).mean()
df['std'] = df['Adj_Close'].rolling(window=100).std()

#create new features from existing data like calculating daily percent change
df['pct_change']= (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Close'] * 100

# re-define all the features you need
df = df[['100ma', 'std', 'Adj_Close', 'Volume', 'pct_change']]


# plotting two graphs in one window...
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

# plot the Adj. close, 100ma, volume
ax1.plot(df.index, df['Adj_Close'])
ax1.plot(df.index, df['std'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

# display legend
ax1.legend()

# display the graph
plt.show()

# define your trading strategy...
# buy: If price > 2% in 5 days
# Sell: If price < 2% in 5 days
# Hold: If price is going to decrease or increase by less than 2% in 5 days


forecast_col = 'Adj_Close'
df.fillna(-99999, inplace=True)

# define number of days out
forecast_out = 30

#create the forecast column
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# drop your forecasting column, if you keep this it's cheating!
x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# preprocess your data and prepare features as x and labels as y. Common preprocessing techniques are minmax, mean, or std deviation (recommended)
x = preprocessing.scale(x)
y = np.array(df['label'])

# apply Principle Component Analysis to manipulate raw features into new ones. Only useful if you have a lot of data that is "potentially" correlated
pca = PCA(n_components=5)
pca.fit(x)
print("PAC values:", pca.explained_variance_ratio_)

# split your data in two groups: train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# apply your classifier: Linear regression... 
clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = round(clf.score(x_test, y_test) * 100, 3)
print ("Linear regression accuracy is", accuracy,"%")


# print cosmetics...
print(len(x), len(y))
print(df.tail())


