#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for rough kmeans clustering class using UCI iris data set

@author Michael Tompkins
@copyright 2016
"""

# Externals
import time
import json
from sys import argv
from collections import Counter
from copy import deepcopy

# Package level imports from /code
from code import RoughKMeans

# Load data from file
data_file = "iris_dataset.json"
dfile = open(data_file, "r")
data2 = json.load(dfile)  # Initial independent variable dataset
dfile.close()
data_key = 'data_set'
resp_key = "response"

print "Counts", Counter(data2["response"])

# Set string targets as integers
target_types = Counter(data2[resp_key]).keys()
print "Target types",target_types
targets = [target_types.index(i) for i in data2[resp_key]]
num_users = len(targets)
print "Determining similarity for :", num_users, " Instances"

# Determine groups for known targets
list0 = [i for i in range(len(targets)) if targets[i] == 0]
list1 = [i for i in range(len(targets)) if targets[i] == 1]
list2 = [i for i in range(len(targets)) if targets[i] == 2]

# Run rough K means
t2 = time.time()
clstrk = RoughKMeans(data2["data_set"],3,wght_lower=0.7,wght_upper=0.3,threshold=1.01)
clstrk.get_rough_clusters()
t3 = time.time()

print "Rough Kmeans Clustering Took: ",t3-t2," secs"
for i in range(clstrk.max_clusters):
    clt1 = str(i)
    print "GROUP",clt1
    print "Totals Group",clt1,len(clstrk.clusters[clt1]["lower"]),len(clstrk.clusters[clt1]["upper"])

    print "Lower vs Target 0",len(set(clstrk.clusters[clt1]["lower"]).intersection(set(list0)))
    print "Lower vs Target 1",len(set(clstrk.clusters[clt1]["lower"]).intersection(set(list1)))
    print "Lower vs Target 2",len(set(clstrk.clusters[clt1]["lower"]).intersection(set(list2)))

    print "Upper vs Target 0",len(set(clstrk.clusters[clt1]["upper"]).intersection(set(list0)))
    print "Upper vs Target 1",len(set(clstrk.clusters[clt1]["upper"]).intersection(set(list2)))
    print "Upper vs Target 2", len(set(clstrk.clusters[clt1]["upper"]).intersection(set(list2)))
