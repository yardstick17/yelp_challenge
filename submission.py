import mapping_business_2_photos as bizz_to_photo
import business_to_label
import os
import numpy as np
from random import randint
from sklearn.externals import joblib
import pickle
import pandas as pd

model = {}
TOTAL_LABEL = 9


def process_it(P, cutoff):
    result = []
    for el in P:
        if el > cutoff:
            result.append(1)
        else:
            result.append(-1)
    return result


def batch_predict(bags, business_id):
    D = {}

    for bus_id in business_id:
        D[bus_id] = list()

    sum_ = [0.0 for i in range(len(bags))]

    for i in range(9):
        sum_ = [0.0 for x in range(len(bags))]

        for j in range(1):
            cutoff = new_cut_off[i][0]
            P = model[j, i].predict(bags)
            t_ = process_it(P, cutoff)
            tmp = [s + p for (s, p) in zip(sum_, t_)]
            sum_ = tmp
        itr = 0
        print sum_
        for bus_id in D:
            if sum_[itr] > 0:
                D[bus_id].append(1)
            else:
                D[bus_id].append(-1)
            itr += 1

    for bus_id in D:
        #print D[bus_id]
        #print 'Length of D[bus_id] : ' , len(D[bus_id])
        file_name = 'result/' + bus_id
        file_ = open(file_name, 'wb')
        pickle.dump(D[bus_id], file_)
        file_.close()

    print 'Batch_Done...'


def batch_submission():
    count = 0
    b_id, new_bags = [], []
    batch = 0
    bags = joblib.load('test_one_bag/bags')
    business_id = joblib.load('test_one_bag/bus_id_list')
    length = len(business_id)
    for bus_id, bag in zip(business_id, bags):
        count += 1

        file_name = 'result/' + bus_id
        if os.path.isfile(file_name) and bus_id != 'business_id':
            individual_business_bag = bag
            b_id.append(bus_id)
            new_bags.append(individual_business_bag)

            print 'Processing : ', batch, ' : ', count
            batch += 1
            del individual_business_bag
        else:
            print 'Already there...', count

        if batch == 100:
            print 'remaining...', length - count
            batch = 0
            batch_predict(new_bags, b_id)
            print b_id
            b_id = []
            new_bags = []

    if batch != 0:
        batch_predict(new_bags, b_id)


for i in range(1):
    for j in range(TOTAL_LABEL):
        print 'Reading models for ', i, j
        file_ = open('models/' + str(i) + str(j) + '.pkl', 'rb')
        model[i, j] = pickle.load(file_)
        file_.close()

for j in range(1):
    file_ = open('models/' + str(j) + '.pkl', 'rb')
    new_cut_off = pickle.load(file_)
    file_.close()

print 'Models read..'


def make_file_for_submission():
    f = open('new_submit.txt', 'wb')
    f.write('business_id,labels\n')
    business_id = joblib.load('test_one_bag/bus_id_list')
    for bus_id in business_id:
        file_name = 'result/' + bus_id
        file_ = open(file_name, 'rb')
        binary_tag = pickle.load(file_)
        file_.close()
        s = ''
        for i in range(9):
            if binary_tag[i] > 0:
                s += str(i) + ' '

        s.strip()
        s += '\n'
        f.write(bus_id + ',' + s)
    f.close()


def make_submission():
    df = pd.DataFrame(columns=['business_id', 'labels'])
    business_id = joblib.load('test_one_bag/bus_id_list')
    index = 0
    for bus_id in business_id:
        file_name = 'result/' + bus_id
        file_ = open(file_name, 'rb')
        binary_tag = pickle.load(file_)
        file_.close()
        s = ''
        for i in range(9):
            if binary_tag[i] > 0:
                s += str(i) + ' '

            s.strip()

        label = s
        df.loc[index] = [str(bus_id), label]
        index += 1

    with open("submission_fc7.csv", 'w') as f:
        df.to_csv(f, index=False)


if __name__ == '__main__':
    batch_submission()
    #make_file_for_submission()
    make_submission()
    print 'Excited?'
