#!/usr/bin/env python2.7
# encoding: utf-8

"""
An implementation of rough clustering for multi-dimensional integer features

@options
max_clusters - integer corresponding to number of clusters to return
max_d - Maximum inter-entity distance to consider before stopping further clustering
if max_d is not specified, then algorithm determines max_d based on inter-entity distance statistics (25th percentile)

@note
The algorithm determines the optimal distance D for final clustering based on option 'objective' which maximizes :
    "lower" : sum of lower approximations - maximum entity uniqueness across all clusters at distance D
    "coverage" : total # of entites covered by all clusters - maximum number of entities across all clusters at distance D
    "ratio" : ratio of lower/coverage (default) - maximum ratio of unique entities to total entities across all clusters at distance D
    "all" : return clusters at every distance D from [0 - self.total_entities]

@author Michael Tompkins
@copyright 2016
"""

# Externals
import time
import itertools
import operator
from collections import Counter
import numpy as npy
from copy import deepcopy


class RoughCluster:

    def __init__(self,input_data,max_clusters,objective="ratio",max_d=None):

        # Rough set clustering output vars
        self.data = input_data
        self.distance = {}
        self.all_keys = {}
        self.clusters = []
        self.sum_upper = []
        self.sum_lower = []
        self.cluster_list = []
        self.total_entities = 0
        self.pruned = {}
        self.optimal = {}
        self.opt_d = None

        self.debug = False
        self.small = 1.0e-10

        # Rough set clustering options
        self.minD = None					# Minimum inter-entity distance to perform clustering over
        self.maxD = max_d					# Maximum inter-entity distance to perform clustering over
        self.objective = objective			# Objective to maximize for optimal clustering distance D
        self.max_clusters = [max_clusters]	# Number of clusters to return

    def get_entity_distances(self):

        """
        Compute inter-entity distance matrix for all unique entities in input

        :var self.data
        :return: self.distance : inter-entity distances for all unique (lower traingular) pairs of entities
        :return self.all_keys
        :return self.total_entities
        :return self.minD
        :return self.maxD (if maxD = None from init method)
        """

        header = self.data.keys()
        data_length = len(self.data[header[0]])

        t1 = time.time()

        # Enumerate entire distance matrix
        for k in range(0,data_length):
            self.distance[str(k)] = {str(j): sum([abs(self.data[val][k]-self.data[val][j]) for val in header])
                                        for j in range(0,data_length)}

        curr_dists = list(itertools.chain([self.distance[h][g] for h in self.distance for g in self.distance[h]]))
        self.minD = int(max([npy.percentile(curr_dists,2),2]))
        if self.maxD is None:   # Determine maxD based on 25th percentile of all inter-cluster distances
            self.maxD = int(max([npy.percentile(curr_dists,25),3]))

        self.all_keys = {str(key): None for key in range(0,data_length)}    # Static all entity keys
        curr_keys = {str(key): None for key in range(0,data_length)}		# Place holder entity keys
        self.total_entities = len(curr_keys.keys())

        # Compute distance of all pairs (p,q) where p != q
        for k in range(0,data_length):
            key1 = str(k)
            curr_keys.pop(key1)
            #self.distance[key1] = {key2 : int(sum([abs(self.data[val][k]-self.data[val][int(key2)]) for val in header]))
            #					   for key2 in curr_keys.keys()}
            self.distance[key1] = {key2 : int(self.distance[key1][key2]) for key2 in curr_keys.keys()}

        t2 = time.time()
        if self.debug is True:
            print "time",t2-t1
            print "Total Entities to Cluster:", self.total_entities
            print "Input Feature Length",len(header)
            print "Max Intra-Entity Distance to Cluster:",self.maxD
            print "Min Intra-Entity Distance to Cluster:",self.minD

        return

    def enumerate_clusters(self):

        """
        Method to enumerate rough clusters given distance measure between all pairs of input entities

        :var self.distance
        :var self.all_keys
        :return : self.sum_lower - lower approximation for each cluster at each distance D
        :return : self.sum_upper - upper approximation for each cluster at each distance D
        :return : self.cluster_list - list of all entities in clusters at each distance D
        :return : self.clusters - list of clusters at each distance D
        """

        # Loop over inter-entity distance D from 0:maxD and find candidate pairs with distance < i
        for i in range(0,self.maxD):
            ct2 = 0
            cluster_count = 0
            cluster_list = []
            clusters = {}
            first_cluster = {}
            # Find entity pairs that have distance <= i
            candidates = {key1:[key2 for key2 in self.distance[key1].keys() if self.distance[key1][key2] <= i ]
                          for key1 in self.all_keys}
            if self.debug is True:
                print "# Candidate Pairs",i,len(list(itertools.chain(*[candidates[g] for g in candidates.keys()]))) #,max(candidates),min(candidates),npy.mean(candidates)
            # Determine for all pairs if pairs are to be assigned to new clusters or previous clusters
            for k,keyname in enumerate(candidates.keys()):
                for l,keyname2 in enumerate(candidates[keyname]):
                    if (keyname in cluster_list) and (keyname2 in cluster_list):	# Assign each entity to other's first cluster
                        if keyname not in clusters[first_cluster[keyname2]]:
                            clusters[first_cluster[keyname2]].append(keyname)
                        if keyname2 not in clusters[first_cluster[keyname]]:
                            clusters[first_cluster[keyname]].append(keyname2)
                            ct2 += 1
                    elif (keyname in cluster_list) and (keyname2 not in cluster_list):	# Assign entity 2 to entity 1's first cluster
                        clusters[first_cluster[keyname]].append(keyname2)
                        cluster_list.append(keyname2)
                        first_cluster[keyname2] = first_cluster[keyname]
                    elif keyname2 in cluster_list and (keyname not in cluster_list):	# Assign entity 1 to entity 2's first cluster
                        clusters[first_cluster[keyname2]].append(keyname)
                        cluster_list.append(keyname)
                        first_cluster[keyname] = first_cluster[keyname2]
                    else:														# Assign both entities to new cluster list
                        clusters[cluster_count] = [keyname,keyname2]
                        cluster_list.append(keyname)
                        cluster_list.append(keyname2)
                        first_cluster[keyname] = cluster_count					# Keep track of current cluster for each key
                        first_cluster[keyname2] = cluster_count					# Keep track of current cluster for each key
                        cluster_count += 1

            if self.debug is True:
                print "Number of Clusters for maxD: ",i," : ",cluster_count

            # Determine upper and lower approximations of clusters for total clusters and pruned clusters
            sum_all = len(list(itertools.chain(*[clusters[g] for g in clusters.keys() if clusters])))
            sum_lower = 0
            sum_upper = 0
            intersections = {}
            int_tmp = {}
            if len(clusters.keys()) > 1:
                for key1 in clusters:
                    intersections[key1] = {key2 : list(set(clusters[key1]).intersection(set(clusters[key2])))
                                     for key2 in clusters if key2 != key1}
                    int_tmp[key1] = len(clusters[key1]) - len(Counter(list(itertools.chain(*[intersections[key1][g]
                                    for g in intersections[key1]]))))
                    sum_lower += int_tmp[key1] #intersections[key1])
                    sum_upper += len(clusters[key1])
            else:
                sum_lower = sum_all
                sum_upper = sum_all

            self.sum_lower.append(sum_lower)
            self.sum_upper.append(sum_upper)
            self.cluster_list.append(cluster_list)
            self.clusters.append(clusters)

        return

    def optimize_clusters(self):

        """
        Maximize objective over all distances D [self.minD : self.maxD] to determine optimal distance clustering

        :var self.pruned
        :var self.minD
        :var self.maxD
        :return: self.opt_d : optimal integer distance D
        """

        if self.objective == "lower":
            lst = {h : list(itertools.chain([self.pruned[h]["sum_lower"][g] for g in self.pruned[h]["sum_lower"]]))
                   for h in self.pruned if int(h) >= self.minD}
            sort_lst = sorted(lst.iteritems(), key=operator.itemgetter(1),reverse=True)
            self.opt_d = sort_lst[0][0]

        elif self.objective == "coverage":
            lst = {h : list(itertools.chain([self.pruned[h]["sum_upper"][g] for g in self.pruned[h]["sum_upper"].keys()]))
                   for h in self.pruned if int(h) >= self.minD}
            sort_lst = sorted(lst.iteritems(), key=operator.itemgetter(1),reverse=True)
            self.opt_d = sort_lst[0][0]

        elif self.objective == "ratio":
            lst = {h : list(itertools.chain([float(self.pruned[h]["sum_lower"][g])/
                                             self.pruned[h]["sum_upper"][g] for g in self.pruned[h]["sum_upper"].keys()])) for h in self.pruned
                   if int(h) >= self.minD}
            sort_lst = sorted(lst.iteritems(), key=operator.itemgetter(1),reverse=True)
            self.opt_d = sort_lst[0][0]
        else:
            self.opt_d = self.maxD

        return

    def prune_clusters(self,optimize=False,cluster_name=0):

        """
        Prune all maxD clusters to number of clusters specified in self.max_clusters and associated rough clusters
        from all maxD clusters returned by enumerate_clusters()

        :arg (optional) cluster_name : if supplied only run for given cluster_name key in self.clusters
        :var self.clusters : dictionary return of enumerate_clusters() containing rough clusters and upper/lower approximation sums
        :var self.total_entities : total number of entities to be clustered
        :return pruned : dictionary containing N clusters that maximize upper approximation
        """

        if cluster_name != 0:
            clusters_in = [self.clusters[cluster_name]]
        else:
            clusters_in = deepcopy(self.clusters)

        for q,clusters in enumerate(clusters_in):
            self.pruned[q+cluster_name] = {"cluster_num":{},"sum_lower":{},"sum_upper":{},"percent_covered":{},"cluster_list":{}}
            cluster_upper_approx = {g : len(clusters[g]) for g in clusters}
            tmpmem = sorted(cluster_upper_approx.iteritems(), key=operator.itemgetter(1),reverse=True)
            clusters1 = []
            cluster_count1 = []
            cluster_list1 = []
            for p,value in enumerate(self.max_clusters):
                sorted_clusters = [t[0] for t in tmpmem[0:self.max_clusters[p]]]
                clusters1.append({key : clusters[key] for key in sorted_clusters})
                cluster_count1.append(len(clusters1[p].keys()))
                cluster_list1.append(list(itertools.chain(*[clusters1[p][g] for g in clusters1[p].keys()])))
                # Compute upper/lower approximations for pruned clusters
                sum_all_1 = len(list(itertools.chain(*[clusters1[p][g] for g in clusters1[p].keys() if clusters1])))
                sum_lower1 = 0
                sum_upper1 = 0
                intersections1 = {}
                if len(clusters1[p].keys()) > 1:
                    for key1 in clusters1[p]:
                        intersections1[key1] = {key2 : list(set(clusters1[p][key1]).intersection(set(clusters1[p][key2])))
                                         for key2 in clusters1[p] if key2 != key1}

                        int_tmp1 = len(Counter(list(itertools.chain(*[intersections1[key1][g] for g in intersections1[key1]]))))
                        sum_lower1 += (len(clusters1[p][key1]) - int_tmp1) #intersections[key1])
                        sum_upper1 += len(clusters1[p][key1])
                else:
                    sum_lower1 = sum_all_1
                    sum_upper1 = sum_all_1

                if self.debug is True:
                    print "Intra-Entity Distance : ",q
                    print "Results for : ",self.max_clusters[p]," Pruned Clusters"
                    print "Sum of Lower Approximation for Pruned Clusters :",sum_lower1
                    print "Sum of Upper Approximations for Pruned Clusters",sum_upper1
                    print "Number of Entities Covered for Pruned Clusters",len(Counter(cluster_list1[p]).keys())
                    print "Percentage of Entities Covered for Pruned Clusters", \
                        (len(Counter(cluster_list1[p]).keys())/float(self.total_entities))*100.0

                # Pack stats into output
                self.pruned[q+cluster_name]["cluster_list"][value] = clusters1[p]
                self.pruned[q+cluster_name]["cluster_num"][value] = cluster_count1[p]
                self.pruned[q+cluster_name]["sum_lower"][value] = sum_lower1
                self.pruned[q+cluster_name]["sum_upper"][value] = sum_upper1
                self.pruned[q+cluster_name]["percent_covered"][value] = (len(Counter(cluster_list1[p]).keys())/float(self.total_entities))*100.0

        # Find optimal distance D cluster based on self.objective
        if optimize is True:
            self.optimize_clusters()
            self.optimal = {self.opt_d : self.pruned[self.opt_d]}

        return

if __name__ == "__main__":

    """
    For class-level tests see /tests/rough_clustering_tests.py
    """