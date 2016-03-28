import mapping_business_2_photos as bizz_to_photo
import business_to_label
import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from operator import add
business_id = bizz_to_photo.mapping_biz_to_photos()
import pickle
from sklearn.externals import joblib
bizz_to_label = business_to_label.biz_to_label()

scaler = joblib.load('Files/preprocessing/scaler_new.pkl')
pca = joblib.load('Files/preprocessing/pca.pkl')

def individual_business_bag_extract(bus_id):
	individual_business_bag = []
	photo_id = business_id[bus_id]
	for id in photo_id:
		try:
			row = []
			with open('/home/ubuntu/caffe/python/yelp/train_photos_feature/' + str(id) ) as f:
				content =  f.read()
			str_ = str(content)
			str_ = str_[1:-1]
			new_ = str_.replace('\n' , '').replace('  ', ' ').split('  ')
			for e in new_:
				row.append(float(e.strip())*10000)
			individual_business_bag.append(np.array(row))
			#print 'File_Found..'
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
			file_name = 'train_bags_prob_layer/' + bus_id
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
	labels = []
	p =''
	count = 0
	for bus_id in business_id:
		print 'total.. : ' , count
		count += 1 
		file_name = 'train_bags_prob_layer/' + bus_id
		if bus_id != 'business_id':
			bag = joblib.load(file_name)
			bag = scaler.transform(bag)
			bag = pca.transform(bag)
			bags.append(bag)
			labels.append(bizz_to_label[bus_id])
	joblib.dump(bags , 'train_one_bag/bags')
	joblib.dump(labels , 'train_one_bag/labels')
			
	
if __name__ == '__main__':
	#prepare_for_training() 
	prepare_one_bag()
	print 'BAGS popultated :)'

