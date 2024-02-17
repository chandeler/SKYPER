# coding: utf-8

import json


with open("data_origin/train.txt", "r") as reader:
    train = [d.strip() for d in reader.readlines()]
with open("data_origin/test.txt", "r") as reader:
    test = [d.strip() for d in reader.readlines()]
with open("data_origin/valid.txt", "r") as reader:
    valid = [d.strip() for d in reader.readlines()]

for i in range(5):
    query = ["queryid" + str(q.get('ridx')) for q in json.load(open("query_%d.json" % i, "r"))]
    test_lst = []
    train_lst = []
    for q in test:
        if q.split("\t")[0] in query:
            test_lst.append(q)
        else:
            train_lst.append(q)
    for d in train:
        train_lst.append(d)
    with open("data_5fold/query_%d/train.txt" % i, "w") as writer:
        [writer.write(d + "\n") for d in train_lst]
    with open("data_5fold/query_%d/test.txt" % i, "w") as writer:
        [writer.write(d + "\n") for d in test_lst]
    with open("data_5fold/query_%d/valid.txt" % i, "w") as writer:
        [writer.write(d + "\n") for d in valid]
