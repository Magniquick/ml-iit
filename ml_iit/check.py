import csv
types = dict()
for i in ("test.csv", "train.csv", "valid.csv"):
	with open(i, newline='') as f:
		reader = csv.reader(f, delimiter=',')
		reader.__next__()
		for row in reader:
			if len(row) != 2:
				print(row)
			if type(row[0]) != str:
				print(row)

			if row[-1] not in types:
				types[row[-1]] = 1
			else:
				types[row[-1]] += 1
	print(i, types)
	types = dict()
