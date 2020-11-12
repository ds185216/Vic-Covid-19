import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv('CasesGSG_data.csv').iloc[::-1].fillna(value=0)
for frame in range(99):
	vic=data[-100:]
	if frame != 0:
		vic = vic[:-frame]
	day = [g for g in vic['Day of Date']][-1]
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
	y_plot['ML Predicted 14 day average'] = y_plot['combined'].iloc[:].rolling(window=14).mean()
	y_plot = y_plot.drop(columns = ['combined', 'Predicted Cases'])
	y_plot['ML Predicted 14 day average'][:len(vic)-1] = np.nan
	y_plot = y_plot[14:]
	#fig = plt.figure(figsize=(7,15)) #Vertical
	fig = plt.figure(figsize=(12,7)) #Horizontal
	plt.plot(y_plot, linewidth=4)
	fig.text(0.8,0.7, 'github.com/DS185216/Vic-Covid-19',fontsize=22, color='gray', ha='right', va='bottom', alpha=0.6,rotation=0)
	fig.text(0.2,0.8, day,fontsize=22, color='red', ha='center', va='bottom', alpha=1.0,rotation=0)
	zero_date = y_plot[:(y_plot['ML Predicted 14 day average'].argmin())]['ML Predicted 14 day average']
	if [x for x in zero_date][-1] == 0:
		zero = '0.0 Cases -> %s' %zero_date.index[-1].strftime('%d/%m/%Y')
		fig.text(0.25,0.7, zero,fontsize=12, color='red', ha='center', va='bottom', alpha=1.0,rotation=0)
	plt.title('Predicted COVID-19 cases for Victoria', fontsize = 20)
	fig.legend(y_plot.columns, loc=5, fontsize=16)
	plt.tight_layout()
	fig.autofmt_xdate()
	#plt.show()
	if frame == 0:
		for x in range(12):
			plt.savefig('Chart%s.png' %str(frame+x))
	else:
		plt.savefig('Chart%s.png' %str(frame+11))