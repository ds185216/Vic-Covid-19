import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from lmfit.models import LorentzianModel
#%matplotlib inline
data = pd.read_csv('Cases_data.csv').iloc[::-1].fillna(value=0)

vic=data[-60:]
vic.index = range(len(vic))
#model = LorentzianModel()
#params = model.guess(vic['Cases'], x=vic.index)
#result = model.Cases(vic['Cases'], params, x=vic.index)
#vic['Cases'] = [x for x in  result.eval()]
#vic = vic[-60:]

vic.index = range(len(vic))

y = pd.DataFrame(vic['Cases'][-14:])
X = pd.DataFrame(vic.index[-14:])
#y = pd.DataFrame(vic['Cases'])
#X = pd.DataFrame(vic.index)


#vic = vic[-100:]
date_index = pd.date_range(start = vic['Day of Date'][1], end = '25-12-20')



#model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model = LinearRegression()
model.fit(X, y)
y_plot = model.predict([[x] for x in range(len(date_index))])
y_plot = pd.DataFrame(y_plot)
y_plot.columns = ['Predicted Cases']
y_plot = y_plot.round({'Predicted Cases' : 0})

y_plot[y_plot['Predicted Cases'] < 0] = 0
#y_plot['Cases'] = vic['Cases']



y_plot['Actual confirmed cases'] = vic['Cases']
y_plot['14 day average'] = vic['Cases'].iloc[:].rolling(window=14).mean()
#y_plot['Predicted 14 day average'] = y_plot['Predicted Cases'].iloc[:].rolling(window=14).mean()

y_plot.index = date_index



#y_plot = y_plot[:(y_plot['Predicted 14 day average'].argmin())+1]


print (y_plot.tail(30))

y_plot['Predicted Cases'][:len(vic)-1] = np.nan
y_plot['combined'] = y_plot['Predicted Cases'].fillna(y_plot['Actual confirmed cases'])
y_plot['Predicted 14 day average'] = y_plot['combined'].iloc[:].rolling(window=14).mean()
y_plot = y_plot.drop(columns = ['combined', 'Predicted Cases'])
y_plot['Predicted 14 day average'][:len(vic)-1] = np.nan
y_plot = y_plot[14:]

graph = y_plot.plot(title = 'Predicted COVID-19 cases for Victoria', figsize=(5,5), linewidth=4)
graph
plt.show()