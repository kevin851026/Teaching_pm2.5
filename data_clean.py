# -*- coding: utf-8 -*-
import csv

with open('pm2.5Taiwan.csv',encoding="utf-8") as input_csv \
		,open('cleaned_pm2.5.csv','w', encoding="utf-8") as output_csv:

	reader = csv.reader(input_csv)  # 讀取 CSV
	writer = csv.writer(output_csv) # 寫入 CSV
	next(reader)
	# 第一行
	writer.writerow(['日期','測站','AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED','WS_HR'])
	x=1
	temp_dic=[]
	row_iter=1
	sum_temp=0
	amount=0
	for row in reader:
		for i in range(3,27):
			try:
				sum_temp+=float(row[i])
				amount+=1
			except:
				continue
		if amount==0:
			temp_dic.append(0)
		else:
			temp_dic.append(round(sum_temp/amount,5))
		row_iter+=1
		sum_temp=0
		amount=0
		if row_iter==15:
			writer.writerow( [ row[0] , row[1] ] + temp_dic)

			temp_dic=[]
			row_iter=1
			print(x)
			x+=1
			# break
	# print(temp_dic)
