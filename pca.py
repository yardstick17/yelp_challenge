import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import glob
import mapping_business_2_photos as bizz_to_photo
import business_to_label
import numpy as np
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from operator import add
business_id = bizz_to_photo.mapping_biz_to_photos()
test_business_id = bizz_to_photo.test_mapping_biz_to_photos()
import pickle
from sklearn.externals import joblib
import glob
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
pca = PCA(n_components=50)


def prepare_bag():
    bags = []
    count = 0
    for bus_id in business_id:
        count += 1
        if bus_id != 'business_id':
            print count
            file_name = 'train_bags_prob_layer/' + bus_id
            bag = joblib.load(file_name)
            for row in bag:
                bags.append(row)
        if count > 250:
            break

    count = 0
    for bus_id in test_business_id:
        count += 1
        try:
            print count
            if bus_id != 'business_id':
                file_name = 'test_bags_prob_layer/' + bus_id
                bag = joblib.load(file_name)
                for row in bag:
                    bags.append(row)
            if count > 250:
                break
        except:
            pass

    return bags


X = prepare_bag()
print 'returned..'

scaler = joblib.load('Files/preprocessing/scaler_new.pkl')
print 'scaling data b4 pca'
X = scaler.transform(X)
print 'Scaled.... now fitting... :)'
pca.fit(X)
print 'Fitteed..'
joblib.dump(pca, 'Files/preprocessing/pca.pkl')
print 'Saved...'
