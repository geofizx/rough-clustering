#!/usr/bin/env python2.7
# encoding: utf-8

"""
Some unit tests and usage examples for rough_clustering class

@data UCI Statlog Data Set:
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA:
University of California, School of Information and Computer Science.

@author Michael Tompkins
@copyright 2016
"""

#Externals
import time
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as npy
from scipy.cluster.vq import kmeans2

from code import RoughCluster,RoughKMeans

# Set some rough clustering parameters
maxD = 20			# if None, maxD will be determined by algorithm
max_clusters = 2	# Number of clusters to return

# Load some data
file2 = open("german_all.json","r")
data = json.load(file2)
print data.keys()

# Do some numerical encoding for input payload
header = []
data2 = {}
for key in data["payload"].keys():
    header.append(key)
    try:
        data2[key] = [int(data["payload"][key][m]) for m in range(0,len(data["payload"][key]))]
        if key == "amount":
            data2[key] = []
            for n in range(len(data["payload"][key])):
                bins = [0,1500,3000,8000,20000]
                for i,val in enumerate(bins[0:-1]):
                    if (int(data["payload"][key][n])) >= val and (int(data["payload"][key][n]) < bins[i+1]):
                        data2[key].append(i+1)
    except:
        data2[key] = []
        encoding = {key : m for m,key in enumerate(Counter(data["payload"][key]).keys())}
        for n in range(len(data["payload"][key])):
            data2[key].append(encoding[data["payload"][key][n]])

# Instantiate and run rough clustering
t1 = time.time()
clust = RoughCluster(data2,max_clusters,"ratio",maxD)
clust.get_entity_distances()
clust.enumerate_clusters()
clust.prune_clusters(optimize=True)
t2 = time.time()
print "Rough Set Clustering Took: ",t2-t1," secs"

# Run rough kmeans as well
clstrk = RoughKMeans(data2,2,0.75,0.25,1.3)
clstrk.get_rough_clusters()
t3 = time.time()
print "Rough Kmeans Clustering Took: ",t3-t2," secs"

# Compare results with known centroid mean and std deviations as well as those from k-means
# Print stats for members of clusters
# Determine labels from known classes for "good" and "bad" credit risk
list1 = [i for i in range(len(data["response"])) if data["response"][i] == '1']
list2 = [i for i in range(len(data["response"])) if data["response"][i] == '2']

tableau_lists = []
tableau_1 = []
tableau_2 = []
for key in header:
    tableau_lists.append(data2[key][:])
    tableau_1.append(npy.asarray(data2[key])[list1].tolist())
    tableau_2.append(npy.asarray(data2[key])[list2].tolist())
datav = npy.asfarray(tableau_lists).T
data1 = npy.asarray(tableau_1).T
data3 = npy.asarray(tableau_2).T

mean1 = npy.mean(data1,axis=0)
mean2 = npy.mean(data3,axis=0)
std1 = npy.std(data1,axis=0)
std2 = npy.std(data3,axis=0)

# Just run k-means to compare centroid mean and std deviations
[centroids,groups] = kmeans2(datav,2,iter=20)
meank = [[] for g in range(2)]
val = [[] for n in range(len(groups))]
for m in range(len(groups)):
    for n in range(2):
        if groups[m] == n:
            val[n].append(data["response"][m])
            meank[n].append(datav[int(val[n][-1]),:])
meankp = []
stddevk = []
for n in range(2):
    meankp.append(npy.mean(meank[n],axis=0))
    stddevk.append(npy.std(meank[n],axis=0))

# Compile stats for rough clusters and plot results with known class statistics and kmeans results
resultsm = []
resultss = []
rangek = [l+.2 for l in range(20)]
ranger = [l+.1 for l in range(20)]
print "total instances",Counter(data["response"])

key1 = clust.opt_d 	# Optimal distance D to plot
fig, axs = plt.subplots(nrows=1,ncols=1)
axs.errorbar(range(20),mean2,fmt='ro',yerr=std2,label="True Good Mean+-Std Dev")
plt.hold(True)
plt.title("Comparison of Mean +- Std Deviation for 'Good' and 'Bad' Credit Risk Clustering")
axs.errorbar(range(20),mean1,fmt='bo',yerr=std1,label="True Bad Mean+-Std Dev")
axs.errorbar(rangek,meankp[1],fmt='r+',yerr=stddevk[0],label="Kmeans 0 Class")
axs.errorbar(rangek,meankp[0],fmt='b+',yerr=stddevk[1],label="Kmeans 1 Class")
print "Optimal Clusters",clust.pruned[key1]
print "Optimal Intra-Entity Distance",clust.opt_d
markers = ['bv','rv','gv','kv']
ct = 0
for key in clust.pruned[key1]["cluster_list"][max_clusters]:
    datav2 = []
    meant = []
    stdt = []
    for val in clust.pruned[key1]["cluster_list"][max_clusters][key]:
        meant.append(data["response"][int(val)])
        datav2.append(datav[int(val),:])
    tmp = npy.mean(npy.asarray(datav2),axis=0)
    tmp2 = npy.std(npy.asarray(datav2),axis=0)
    axs.errorbar(ranger,tmp,fmt=markers[ct],yerr=tmp2,label="Rough Cluster D="+str(key1)+" Cluster:"+str(key))
    resultsm.append(npy.mean(npy.asarray(datav2),axis=0))
    resultss.append(npy.std((npy.asarray(datav2)),axis=0))
    print key1,key,len(meant),Counter(meant)
    ct += 1
print "kmeans groups",len([i for i in groups if i == 0]),len([i for i in groups if i == 1])
print "Rough kmeans groups",len(clstrk.groups['0']),len(clstrk.groups['1'])
print "Cluster 0 vs Target 0",len(set(clstrk.groups['0']).intersection(set(list1)))
print "Cluster 1 vs Target 1",len(set(clstrk.groups['1']).intersection(set(list2)))
print "Cluster 0 vs Target 1",len(set(clstrk.groups['0']).intersection(set(list2)))
print "Cluster 1 vs Target 0",len(set(clstrk.groups['1']).intersection(set(list1)))

plt.axis([0,20,-2,7])
plt.xlabel("Feature Number",fontsize=14)
plt.ylabel("Centroid Mean +- Std Dev.",fontsize=14)
plt.legend()
plt.show()
