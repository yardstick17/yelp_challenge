import mapping_business_2_photos as bizz_to_photo
import business_to_label
import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from operator import add
business_id = bizz_to_photo.test_mapping_biz_to_photos()
import pickle
from sklearn.externals import joblib
scaler = joblib.load('Files/preprocessing/scaler_new.pkl')
pca = joblib.load('Files/preprocessing/pca.pkl')

def individual_business_bag_extract(bus_id):
	individual_business_bag = []
	photo_id = business_id[bus_id]
	for id in photo_id:
		try:
			row = []
			with open('/home/ubuntu/caffe/python/yelp/test_photos_feature/' + str(id) ) as f:
				content =  f.read()
			str_ = str(content)
			str_ = str_[1:-1]
			new_ = str_.replace('\n' , '').replace('  ', ' ').split('  ')
			for e in new_:
				row.append(float(e.strip()) * 10000)
			individual_business_bag.append(np.array(row))
		except:
			print 'File not Found'
			print id
			raw_input('Hey !! u got a problem..') 
			pass

	del photo_id
	print 'total pic in this business_id : ' , len(individual_business_bag)
	return (individual_business_bag)

def prepare_for_training():
	count = 0
	for bus_id in business_id:
		if bus_id != 'business_id':
			file_name = 'test_bags_prob_layer/' + bus_id
			count += 1
			if not os.path.isfile(file_name):
				try:
					individual_business_bag = individual_business_bag_extract(bus_id)
					joblib.dump( individual_business_bag , file_name)
					print 'File for ' , bus_id , 'written on disk.. : ' , count
					del individual_business_bag
				except:
					print 'Hua chutiyap.....'
			else:
				print 'Already read!!'


def prepare_one_bag():
	bags = []
	bus_id_list = []
	count = 0
	for bus_id in business_id:
		file_name = 'test_bags_prob_layer/' + bus_id
		if bus_id != 'business_id':
			print 'Doing..' , bus_id , count
			count += 1
			bag = joblib.load(file_name)
			bag = scaler.transform(bag)
			bag = pca.transform(bag)
			bags.append(bag)
			bus_id_list.append(bus_id)
	joblib.dump(bags , 'test_one_bag/bags')
	joblib.dump(bus_id_list , 'test_one_bag/bus_id_list')

if __name__ == '__main__':
	#prepare_for_training() 
	prepare_one_bag()
	print 'BAGS popultated :)'

