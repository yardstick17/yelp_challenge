from sklearn.externals import joblib
import numpy as np
TL  = joblib.load('TL/TL')
PL = joblib.load('PL/PL')
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

T = (list(TL))
P = list(PL)
T = []
P = []
for row in TL:
	new_row = []
	for el in row:
		if el < 0:
			new_row.append(0)
		else:
			new_row.append(1)
	T.append(new_row)
for row in PL:
	new_row = []
	for el in row:
		if el < 0:
			new_row.append(0)
		else:
			new_row.append(1)	
	P.append(new_row)
p = np.array(P)
t = np.array(T)
i = 0

res = []
for row in p:
	r = []
	for i in range(9):
		if row[i]>0:
			r.append(str(i))
	res.append(r)

tes = []
for row in t:
	r = []
	for i in range(9):
		if row[i]>0:
			r.append(str(i))
	tes.append(r)


for x , y in zip(tes ,res):
	print x
	print y
	print '\n'
smp_report = precision_recall_fscore_support(p , t , average = 'samples')
print "F1 score: ", f1_score(p, t, average='micro') 

print  smp_report
