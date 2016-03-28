import mapping_business_2_photos as bizz_to_photo
import business_to_label
import numpy as np 
import os
from sklearn.externals import joblib
import sklearn.preprocessing
from sklearn import preprocessing
from operator import add
business_id = bizz_to_photo.mapping_biz_to_photos()
test_business_id = bizz_to_photo.test_mapping_biz_to_photos()
import pickle
def prepare_bag():
	bags = []
	count = 0
	for bus_id in business_id:
		count += 1
		if bus_id != 'business_id':
			file_name = 'train_bags_prob_layer/' + bus_id
			bag = joblib.load(file_name)
			for row in bag:
				bags.append(row)
		if count > 500:
			break
	
	count = 0
	for bus_id in test_business_id:
		try:
			count += 1
			if bus_id != 'business_id':
				file_name = 'test_bags_prob_layer/' + bus_id
				bag = joblib.load(file_name)
				for row in bag:
					bags.append(row)
			if count > 500:
				break
		except:
			pass


	return bags

X = prepare_bag()
print 'returned..'
scaler = preprocessing.StandardScaler().fit(X)
print 'Fitted...'
joblib.dump(scaler , 'Files/preprocessing/scaler_new.pkl')
print 'go ahead....'