# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# import tensorflow as tf
import csv
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
with open('cleaned_pm2.5.csv',encoding="utf-8") as csvfile:
	# 每一行長這樣
	# ['日期','測站','AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
	#    , 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED','WS_HR']
	reader = csv.reader(csvfile)  # 讀取 CSV
	field=['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10'
	   		,'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED','WS_HR']
	data_array=[]
	for i in range(14):
		data_array.append([])
	x=1
	next(reader)#跳過第一行
	next(reader)#不知道為何都會讀到一行[]
	for row in reader:
		# print(row)
		for i in range(2,16):
			data_array[i-2].append(float(row[i]))
		next(reader)
		# if x==10:
		# 	break
		x+=1
	pm25=data_array[7]
	del(data_array[7])
	allchart=[]
	for i in range(10):
		# print(data_array[i])
		# print(pm25)
		mean=np.mean(data_array[i])
		std=np.std(data_array[i])
		allchart.append(np.array(data_array[i]))
		temp_data=[]
		temp_pm25=[]
		for j ,k in zip(data_array[i],pm25):
			if j < mean+3*std and j > mean-3*std :
				temp_data.append(j)
				temp_pm25.append(k)
		# ----------------將訓練資料畫至XY平面-----------------------
		plt.scatter(temp_data, temp_pm25);           #是否歸一到0~1之間
		# plt.scatter(softmax(temp_data), temp_pm25);

		# -------------------預測模型訓練-------------------------
		print(temp_pm25[0])
		exit()
		# model = LinearRegression(fit_intercept=True) 		#線性模型
		# temp_data=np.array(temp_data)						#轉換訓練資料型態
		# print('start')
		# model.fit(temp_data[:, np.newaxis], temp_pm25)	#訓練開始
		# print('end')
		# xfit = np.linspace(min(temp_data), max(temp_data), 1000) 	#在X軸上sample 1000個點
		# yfit = model.predict(xfit[:, np.newaxis])					#預測結果
		# plt.plot(xfit, yfit,'r');									#將結果線畫至圖上
		## plt.show()
		# plt.savefig('./chart/'+field[i]+'.jpg')
		# plt.close()
		print(i)
	allchart=np.array(allchart)
	x_axis = np.linspace(0 , 100 , 1000)							#sample 1000個點
	xfit = np.tile(x_axis.reshape(-1),(10,1))						#多維平面畫不出來 故將其全部放在xy平面上看
	print('start')
	# ---------------線性模型--------------------------
	# model = LinearRegression(fit_intercept=True)
	# model.fit(allchart.T, pm25)
	# yfit = model.predict(xfit.T)
	# ---------------多項式模型-----------------------------
	poly_model = make_pipeline(PolynomialFeatures(4), LinearRegression()) 	#設定多項式degree
	poly_model.fit(allchart.T, pm25)										#訓練開始
	yfit = poly_model.predict(xfit.T)										#預測結果
	# -------------------------------------------------------
	print('end')
	plt.plot(x_axis, yfit,'r');
	# plt.plot(softmax(x_axis), yfit,'r');
	plt.show()
	# plt.savefig('./chart/all.jpg')