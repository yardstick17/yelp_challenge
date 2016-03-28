import mapping_business_2_photos as bizz_to_photo
import business_to_label
import os
import numpy as np
from random import randint
from sklearn.externals import joblib
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
model = {}
TOTAL_LABEL = 9

for i in range(1):
	for j in range(TOTAL_LABEL):
		print 'models loading...'  , i , j 
		file_ = open('models/' + str(i) + str(j) + '.pkl' , 'rb')
		model[i,j] = pickle.load(file_)
		file_.close()

file_ = open('models/' + str(0) + '.pkl' , 'rb')
new_cut_off = pickle.load(file_)
file_.close()

def process_it(P , cutoff):
	result = []
	for el in P:
		if el > cutoff:
			result.append(1)
		else:
			result.append(-1)
	return result

def process(labels):
	new_labels = []
	for row in labels:
		row_labels = []
		for i in range(9):
			if str(i) in row:
				row_labels.append(1)
			else:
				row_labels.append(-1)
		new_labels.append(row_labels)
	return new_labels

def batch_predict(bags , business_id , labels):
	D={}
	for bus_id in business_id:
		D[bus_id] = list()
	
	sum_ = [0.0 for x in range(len(bags))]
	
	for i in range(9):
		sum_ = [0.0 for x in range(len(bags))]
	
		for j in range(1):
			if j != 0:
				continue
			cutoff = new_cut_off[i][0]
			P = model[j,i].predict(bags)
			t_ = process_it(P , cutoff )
			tmp = [g + h for g , h in zip(t_ , sum_)]
			sum_ = tmp
		itr = 0
		print sum_
		for bus_id in D:
			if sum_[itr] > 0:
				D[bus_id].append(1)
			else:
				D[bus_id].append(-1)
			itr += 1
	predicted_label = []
	for bus_id in D:
		predicted_label.append(D[bus_id])
	
	test_lablels = np.array(process(labels))
	print 'Batch_Done...'
	joblib.dump(test_lablels , 'Files/TL/TL')
	joblib.dump(np.array(predicted_label) , 'Files/PL/PL')
	print 'Yo'



def batch_submission():
	count = 0
	b_id , new_bags = [] , []
	batch = 0
	
	bags = joblib.load('train_one_bag/bags')
	labels = joblib.load('train_one_bag/labels')
	bags = bags[1100:1300]
	labels = labels[1100:1300]

	business_id = joblib.load('test_one_bag/bus_id_list')
	length = len(business_id)
	for bus_id , bag in zip(business_id , bags):
		count += 1
		
		file_name = 'score/' + bus_id
		if not os.path.isfile(file_name) and  bus_id != 'business_id':
			individual_business_bag = bag
			b_id.append(bus_id)
			new_bags.append(individual_business_bag)
			
			print 'Processing : ' , batch  , ' : ' , count
			batch += 1
			del individual_business_bag
		else:
			print 'Already there...' , count
		
		if batch == 200:
			print 'remaining...' , length - count
			batch = 0
			batch_predict(new_bags ,b_id , labels)
			print b_id
			b_id = []
			new_bags = []

	if batch != 0:
		pass

if __name__ == '__main__':
	batch_submission()

	print 'Excited?'
	
