import csv
from typing import Dict, List

class files:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: Dict[str,int] = {}
        self.writer = csv.writer(open(self.filename, mode='w', newline=''))

    def write_to_csv(self, values: List[str]):
        self.data[values[1]] += 1
        self.writer.writerow(values)

test = files("test.csv")
train = files("train.csv")
valid = files("valid.csv")
all = (test, train, valid)

with open('./dataset/train.csv', newline='') as og:
    og.__next__()
    reader = csv.reader(og, delimiter=',')
    for row in reader:
        skip = False
        for i in all:
            if i.data.setdefault(row[2],0) == 0:
                i.write_to_csv(row[1:])
                skip = True
        if skip:
            continue

        if valid.data[row[-1]]/train.data[row[-1]] < 1/7:
            valid.write_to_csv(row[1:])
        elif test.data[row[-1]]/train.data[row[-1]] < 2/7:
            test.write_to_csv(row[1:])
        else:
            train.write_to_csv(row[1:])