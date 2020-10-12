import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv('Cases_data.csv').iloc[::-1].fillna(value=0)
vic=data[-60:]
vic.index = range(len(vic))
vic.index = range(len(vic))
y = pd.DataFrame(vic['Cases'][-14:])
X = pd.DataFrame(vic.index[-14:])
date_index = pd.date_range(start = vic['Day of Date'][1], end = '25-12-20')
model = LinearRegression()
model.fit(X, y)
y_plot = model.predict([[x] for x in range(len(date_index))])
y_plot = pd.DataFrame(y_plot)
y_plot.columns = ['Predicted Cases']
y_plot = y_plot.round({'Predicted Cases' : 0})
y_plot[y_plot['Predicted Cases'] < 0] = 0
y_plot['Actual confirmed cases'] = vic['Cases']
y_plot['14 day average'] = vic['Cases'].iloc[:].rolling(window=14).mean()
y_plot.index = date_index
y_plot['Predicted Cases'][:len(vic)-1] = np.nan
y_plot['combined'] = y_plot['Predicted Cases'].fillna(y_plot['Actual confirmed cases'])
y_plot['Predicted 14 day average'] = y_plot['combined'].iloc[:].rolling(window=14).mean()
y_plot = y_plot.drop(columns = ['combined', 'Predicted Cases'])
y_plot['Predicted 14 day average'][:len(vic)-1] = np.nan
y_plot = y_plot[14:]
fig = plt.figure(figsize=(5,5))
plt.plot(y_plot, linewidth=3)
fig.text(0.8,0.7, 'github.com/DS185216/Vic-Covid-19',fontsize=13, color='gray', ha='right', va='bottom', alpha=0.6,rotation=0)
plt.title('Predicted COVID-19 cases for Victoria')
fig.legend(y_plot.columns, loc=5)
plt.tight_layout()
fig.autofmt_xdate()
plt.show()