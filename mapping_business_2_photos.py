import csv


def mapping_biz_to_photos():
    business_id = {}
    with open('Files/train_photo_to_biz_ids.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            try:
                business_id[row[1]].append(row[0])  # print row
            except:
                business_id[row[1]] = list()
                business_id[row[1]].append(row[0])
    return business_id


def test_mapping_biz_to_photos():
    test_business_id = {}
    with open('Files/test_photo_to_biz.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            try:
                test_business_id[row[1]].append(row[0])  # print row
            except:
                test_business_id[row[1]] = list()
                test_business_id[row[1]].append(row[0])
    return test_business_id
