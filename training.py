import misvm
import numpy as np 
import mapping_business_2_photos as bizz_to_photo
import business_to_label
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support

business_id = bizz_to_photo.mapping_biz_to_photos()
bizz_to_label = business_to_label.biz_to_label()
TOTAL_LABEL = 9

print 'bags loading...'
bags = joblib.load('train_one_bag/bags')
tags = joblib.load('train_one_bag/labels')

print 'bags loaded...'

def get_labels(tags):
	labels = []
	for row in tags:
		label = []
		for i in range(TOTAL_LABEL):
			if str(i) in row:
				label.append(1.0)
			else:
				label.append(-1.0)
		l = np.array(label , dtype = float)
		labels.append(l)
	return labels

def calculate_mean(test_labels , predictions):
	tp , tpfp , ap = 0, 0 , 0
	for i , j in zip(test_labels , predictions):
		if i == 1:
			if j == 1:
				ap += 1
			tp += 1
		if j == 1:
			tpfp += 1
	try:
		precison = float(ap) / tpfp 
		recall = float(ap) / tp
		mean = (2*precison * recall) / (precison + recall)
	except:
		mean = 0
	print 'mean is : ' , mean
	return mean 

def f1_score_new(test_labels , predictions):
	T  , P = []  , []
	#print test_labels
	#print predictions
	r = []
	for el in test_labels:
		if el > 0:
			r.append(1)
		else:
			r.append(0)
	
	T.append(r)
	
	r = []
	for el in predictions:
		if el > 0:
			r.append(1)
		else:
			r.append(0)
	
	P.append(r)
	x = f1_score(np.array(P), np.array(T) , average='micro')
	smp_report = precision_recall_fscore_support(np.array(P) , np.array(T) , average = 'samples')
	del P
	del T
	del r
	return x  , smp_report

	
def chosse_threshold(predictions , test_labels):
	C = -3.0
	step , threshold = 0.01 , 0.0
	a , max_acc = [] , 0.0
	while C <= 3.0:
		a = []
		for pre in predictions:
			if pre > C:
				a.append(1)
			else:
				a.append(-1)
		P = np.array(a)
		#acc = calculate_mean(test_labels , np.sign(P) )
		acc , report = f1_score_new(test_labels , np.sign(P))

		if max_acc < acc:
			max_acc =  acc
			threshold = C
			final_predictions = np.sign(P)
			final_report = report
		del a 
		C += step
	
	return threshold , max_acc , final_predictions , final_report
	
def get_best_classifier(train_bags ,train_labels , test_bags , test_labels):
    # ***************************   Preparing for Training Phase ********************************* #
	classifiers = {}
	accuracies = {}
	cut_off = {}
	PREDICTOIN = {}
	REPORT = {}

	#cross-check with different set of parameters to best fit our classifier
	C_2d_range = [700 , 900]
	gamma_2d_range = [1]
	for C in C_2d_range:
	    for gamma in gamma_2d_range:
	    	#classifiers['miSVM_' + str(C) + ' ' + str(gamma)] = misvm.MISVM(kernel='linear_fs', gamma = gamma , C=C, max_iters=20)
	    	classifiers['miSVM_' + str(C) ] = misvm.NSK(kernel='linear_fs', gamma = gamma ,  C=C)
 # *********************************** Training Phase ******************************************** #
	
	for algorithm, classifier in classifiers.items():
		classifier.fit(train_bags, train_labels)
		final_classifier = classifier
	
	# ***************************** TESTING PHASE *************************************************** #	

	
	for algorithm , classifier in classifiers.items():	
		predictions = classifier.predict(test_bags)
		cut_off[algorithm] ,  accuracies[algorithm] , PREDICTOIN[algorithm] , REPORT[algorithm] = chosse_threshold(predictions , test_labels)   #calculate_accuracy() #np.average(test_labels == np.sign(predictions))
		
	
	max_accuracy = -1
	final_algorithm = 'Beta!!'
	
	for algorithm, accuracy in accuracies.items():
		print algorithm , ' : '  , accuracy , ' , cutoff : ' ,cut_off[algorithm]
		if 100 * accuracy > max_accuracy:
			max_accuracy = 100*accuracy
			final_classifier = classifiers[algorithm]
			final_cutoff = cut_off[algorithm]
			final_predictions = PREDICTOIN[algorithm]
			final_algorithm = algorithm
			final_report = REPORT[algorithm]
	
	print 'Actual.......:\n' ,np.array(np.sign(test_labels))
	print 'predictions..:\n' , np.array(final_predictions)
	print 'Finest accuracy : ', max_accuracy , 'for cutoff = ' , final_cutoff , '  , Model :' , final_algorithm
	print 'Report : prepcision and recall  : ' , final_report
	del test_labels
	del train_labels
	del train_bags
	del test_bags
	del PREDICTOIN
	del cut_off
	del accuracies
	del classifiers
	return final_classifier , final_cutoff , max_accuracy

def build_labels(i , business_label_1):
	label = []
	for row in business_label_1:
		label.append(row[i])
	
	return np.array(label , dtype = float)

if __name__ == '__main__':
	BATCH = 700
	BASE = 200
	labels = get_labels(tags)

	test_bags   = bags[:BASE]
	test_labels = labels[:BASE]
	cut_off_list = []
	for j in range(1):
		UPPER_LIMIT = min(BASE + BATCH , len(bags))
		train_bags = bags[BASE:UPPER_LIMIT]
		train_labels = labels[BASE:UPPER_LIMIT]
		cut_off_list = []
		for i in range(TOTAL_LABEL):
			print 'Limits : ' , BASE , UPPER_LIMIT , i 
				
			label_train = build_labels(i , train_labels)
			label_test = build_labels(i , test_labels)
			classifier , cut_off , max_accuracy = get_best_classifier(train_bags , label_train ,test_bags , label_test)
			cut_off_list.append( (cut_off , max_accuracy) )
			file_ = open('models/' + str(j) + str(i) + '.pkl' , 'wb')
			pickle.dump( classifier, file_)
			file_.close()
		
		BASE += BATCH
		file_ = open('models/' + str(j) + '.pkl' , 'wb')
		pickle.dump(cut_off_list , file_)
		file_.close()