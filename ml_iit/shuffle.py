import csv
from typing import Dict, List

class files:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: Dict[str,int] = {}
        self.writer = csv.writer(open(self.filename, mode='w', newline=''))
        self.writer.writerow(("text","labels"))
    def write_to_csv(self, values):
        if values[0] == "":
            print("writing" ,values, len(values), type(values[0]) )
        self.data[values[1]] += 1
        self.writer.writerow(values)

test = files("./dataset/test.csv")
train = files("./dataset/train.csv")
valid = files("./dataset/valid.csv")
all = (test, train, valid)

with open('./og.csv', newline='') as og:
    og.__next__()
    reader = csv.reader(og, delimiter=',')
    for row in reader:

        row[2] = {"FALSE":0,"HALF TRUE":1,"MOSTLY FALSE":2,"PARTLY FALSE":3,"MOSTLY TRUE":4}[row[2]]

        data = row[1:]
        if data[0] == "":
            continue

        skip = False
        for i in all:
            if i.data.setdefault(row[2],0) == 0:
                i.write_to_csv(data)
                skip = True
        if skip:
            continue

        if valid.data[row[-1]]/train.data[row[-1]] < 1/7:
            valid.write_to_csv(data)
        elif test.data[row[-1]]/train.data[row[-1]] < 2/7:
            test.write_to_csv(data)
        else:
            train.write_to_csv(data)
