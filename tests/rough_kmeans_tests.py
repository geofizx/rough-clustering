#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for rough_clustering class

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

try:
    # Load data from file
    data_file = argv[1]
    dfile = open(data_file, "r")
    data1 = json.load(dfile)  # Initial independent variable dataset
    dfile.close()
    data_key = 'data_set'

    try:
        response_file = argv[2]
        rfile = open(response_file, "r")
        data1["response"] = json.load(rfile)  # Initial response dataset
        resp_key = "response"
    except:
        resp_key = "response"

    # resp_key2 = "fraud"
    # resp_key2 = "reversed"
    resp_key2 = "training_reversed"

except Exception as e:
    raise Exception('One or more of your input files are absent for unreadable')

print "Counts", Counter(data1["response"][resp_key2]["target"])
topN = 50  # topN most similar users

# Set user_id to be row number
user_id = [k for k in range(len(data1["response"][resp_key2]["target"]))]
fraud_users = [k for k in user_id if data1["response"][resp_key2]["target"][k] == 1]
num_users = len(fraud_users)
print "Determining similarity for :", num_users, " Charge-Back Users"
print "Total Users Compared :", len(user_id)

file6 = open("DPA_output_training_reversed.json", "r")
DPA = json.load(file6)
data2 = {"response": deepcopy(data1["response"][resp_key2]["target"])}
for key1 in DPA["feature_names"]:
    data2[key1] = data1["data_set"][key1]

list1 = [i for i in range(len(data2["response"])) if data2["response"][i] == 0]
list2 = [i for i in range(len(data2["response"])) if data2["response"][i] == 1]

# Run rough K means
t2 = time.time()
clstrk = RoughKMeans(data2,2,0.75,0.25,1.5)
clstrk.get_rough_clusters()
t3 = time.time()
print "Rough Kmeans Clustering Took: ",t3-t2," secs"
print "Totals Group 0",len(clstrk.clusters['0']["lower"]),len(clstrk.clusters['0']["upper"])
print "Totals Group 1",len(clstrk.clusters['1']["lower"]),len(clstrk.clusters['1']["upper"])

print "Lower 0 vs Target 0",len(set(clstrk.clusters['0']["lower"]).intersection(set(list1)))
print "Lower 1 vs Target 1",len(set(clstrk.clusters['1']["lower"]).intersection(set(list2)))
print "Lower 0 vs Target 1",len(set(clstrk.clusters['0']["lower"]).intersection(set(list2)))
print "Lower 1 vs Target 0",len(set(clstrk.clusters['1']["lower"]).intersection(set(list1)))

print "Upper 0 vs Target 0",len(set(clstrk.clusters['0']["upper"]).intersection(set(list1)))
print "Upper 1 vs Target 1",len(set(clstrk.clusters['1']["upper"]).intersection(set(list2)))
print "Upper 0 vs Target 1",len(set(clstrk.clusters['0']["upper"]).intersection(set(list2)))
print "Upper 1 vs Target 0",len(set(clstrk.clusters['1']["upper"]).intersection(set(list1)))