import sys
sys.path.insert(0, '/Users/zomadmin/Projects/caffe/python/')
import os
import numpy as np
import csv
X = []
i = 0
photo_id = {}
buisness_id = {}
attribute = {}
new_attribute = {}


def photo_to_biz_id():
    with open('train_photo_to_biz_ids.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            photo_id[row[0]] = row[1]  # print row
    return photo_id


def biz_to_label():
    with open('Files/train.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x = []
            list_row = list(row[1:])
            for element in list_row:
                if element == '':
                    print element, row
                    pass
                else:
                    x.append(element)

            try:
                buisness_id[row[0]] = x

            except:
                buisness_id[row[0]] = list()
                buisness_id[row[0]] = x
    return buisness_id


'''
['1000', '1', '2', '3', '4', '5', '6', '7']

['1001', '0', '1', '6', '8']

['100', '1', '2', '4', '5', '6', '7']

['1006', '1', '2', '4', '5', '6']

['1010', '0', '6', '8']

['101', '1', '2', '3', '4', '5', '6']

['1011', '2', '3', '5', '6']

['1012', '1', '2', '3', '5', '6']

['1014', '1', '2', '4', '5', '6']

['1015', '1', '5', '6', '7']

['1017', '5', '6', '8']

['1022', '0', '2', '3', '5', '8']

['1024', '0', '1', '2', '3', '4', '5', '6']

['1026', '1', '2', '4', '5', '6', '7']

['1029', '1', '2', '3', '5', '6', '7']

['1030', '0', '3', '5', '8']

['103', '1', '2', '3', '4', '5', '6', '7']

['1031', '6', '8']

['1032', '1', '2', '3', '5', '6', '7']

['1035', '5', '6', '8']

['1036', '1', '2', '4', '5', '6', '7']

['1038', '1', '2', '4', '5', '6', '7']

['1039', '0', '8']

['1040', '0', '2', '6', '7', '8']

['1041', '0', '8']

['1043', '0', '6', '8']

['104', '0', '1', '6', '8']

['1044', '0', '8']

['1046', '0', '6', '8']

['1048', '0', '1', '2', '3', '5', '6']

['1052', '0', '3', '8']

['105', '2', '3', '5', '6']

['1054', '6', '8']
'''
