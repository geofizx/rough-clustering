#!/usr/bin/env python2.7
# encoding: utf-8

"""
An implementation of rough k-means clustering for multi-dimensional numerical features. This extends conventional k-means
to rough set theory by inclusion of upper and lower approximations in both entity-cluster distance measures and
cluster centroid computations.

See: Lingras and Peter, Applying Rough Set Concepts to Clustering, in G. Peters et al. (eds.), Rough Sets: Selected
Methods and Applications in Management and Engineering, Advanced Information and Knowledge Processing,
DOI 10.1007/978-1-4471-2760-4_2, Springer-Verlag London Limited, 2012.

@options
        self.max_clusters = max_clusters	# Number of clusters to return
        self.wght_lower = wght_lower        # Relative weight of lower approximation for each rough cluster centroid
        self.wght_upper = wght_upper        # Relative weight of upper approximation to each rough cluster centroid
        self.dist_threshold = None          # Threshold for clusters to be considered similar distances

@note
self.dist_threshold is determined in method get_entity_centroid_distances()

@author Michael Tompkins
@copyright 2016
"""

#Externals
import time
import itertools
import operator
from collections import Counter
import numpy as np
from copy import deepcopy


class RoughKMeans:

    def __init__(self,input_data,max_clusters,wght_lower=0.75,wght_upper=0.25,threshold=1.25):

        # Clustering vars
        self.data = input_data
        self.feature_names = input_data.keys()
        self.data_length = len(self.data[self.feature_names[0]])
        self.centroids = {}
        self.cluster_list = {}
        self.groups = {}                    # dictionary containing keys for clusters and lists of all entities in key
        self.distance = {}

        # Rough sets vars
        self.all_keys = {}
        self.clusters = None
        self.sum_upper = []
        self.sum_lower = []
        self.total_entities = 0
        self.pruned = {}
        self.optimal = {}
        self.opt_d = None

        # Rough clustering options
        self.max_clusters = max_clusters	# Number of clusters to return
        self.wght_lower = wght_lower        # Relative weight of lower approximation for each rough cluster centroid
        self.wght_upper = wght_upper        # Relative weight of upper approximation to each rough cluster centroid
        self.dist_threshold = threshold     # Threshold for clusters to be considered similar distances (0.0=conventional kmeans)

        # Overhead
        self.debug = False                   # Debug flag for class
        self.debug_assign = False           # Debug flag for assign_cluster_upper_lower_approximation method
        self.debug_dist = False             # Debug flag for get_entity_centroid_distances method
        self.debug_update = False            # Debug flag for update_centroids method
        self.small = 1.0e-04
        self.large = 1.0e+10
        self.tolerance = 1.0e-03            # Tolerance for stopping iterative clustering

    def get_rough_clusters(self):

        """
        Run iterative clustering solver for rough k-means and return max_cluster rough clusters
        :return: self. centroids, self.assignments, self.upper_approximation, self.lower_approximation
        """

        # Get initial random entity clusters
        self.initialize_centroids()

        # Iterate until centroid convergence
        stop_flag = False
        while stop_flag is False:

            # Back-store centroids
            prev_centroids = deepcopy(self.centroids)
            print "Initializing centroids",prev_centroids

            if self.dist_threshold <= 1.0:
                print "Rough distance threshold set <= 1.0 and will produce conventional k-means solution"

            # Get entity-cluster distances
            self.get_entity_centroid_distances()

            # Compute upper and lower approximations
            self.assign_cluster_upper_lower_approximation()

            # Update centroids with upper and lower approximations
            self.update_centroids()

            # Recompute centroid error
            stop_flag = self.get_centroid_convergence(prev_centroids)

        print "Convergence Reached"

        return

    def get_centroid_convergence(self,previous_centroids):

        """
        Convergence test. Determine if centroids have changed, if so, return False, else True
        """

        centroid_error = np.sum([[abs(self.centroids[k][val] - previous_centroids[k][val]) for k in self.centroids]
                                   for val in self.feature_names])

        if self.debug is True:
            print "Centroid change", centroid_error

        if centroid_error <= self.tolerance:
            return True
        else:
            return False

    def initialize_centroids(self):

        """
        Randomly select [self.max_clusters] initial entities as centroids and assign to self.centroids
        :return: self.centroids : current cluster centroids
        """

        # Select max cluster random entities from input and assign as initial cluster centroids
        candidates = np.random.permutation(self.data_length)[0:self.max_clusters]

        if self.debug is True:
            print "Candidates",candidates,self.feature_names,self.data

        self.centroids = {str(k): {v: self.data[v][candidates[k]] for v in self.feature_names} for
                          k in range(self.max_clusters)}

        return

    def update_centroids(self):

        """
        Update rough centroids for all candidate clusters given their upper/lower approximations membership

        Centroids modified for three cases:
            if lower approx == upper approx, return conventional k-means cluster centroid
            elif lower approx == 0 and upper approx != 0, return upper-lower centroid
            else return weighted mean of lower approx centroids and upper-lower centroids

        uses self.data, self.wght_lower, and self.wght_upper

        :return: self.centroids : updated cluster centroids
        """

        for k in self.clusters:

            if len(self.clusters[k]["lower"]) == len(self.clusters[k]["upper"]):
                # Get lower vectors
                lower = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["lower"]])
                self.centroids[str(k)] = {v: np.mean(lower[:,p]) for p, v in enumerate(self.feature_names)}

            elif len(self.clusters[k]["lower"]) == 0 and len(self.clusters[k]["upper"]) != 0:
                # Get upper vectors
                upper = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["upper"]])
                self.centroids[str(k)] = {v: np.mean(upper[:,p]) for p, v in enumerate(self.feature_names)}

            else:
                upper = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["upper"]])
                lower = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["lower"]])
                self.centroids[str(k)] = {v: self.wght_lower*np.mean(lower[:,p],axis=0) +
                                          self.wght_upper*np.mean(upper[:,p],axis=0)
                                          for p, v in enumerate(self.feature_names)}

            if self.debug_update is True:
                print """###Cluster""", k, self.clusters[k]["lower"], self.clusters[k]["upper"]

        return

    def assign_cluster_upper_lower_approximation(self):

        """
        Compute entity-to-cluster assignments + upper/lower approximations for all current clusters

        :return: self.clusters[clusters]["upper"], self.clusters[clusters]["lower"] : for each cluster
        """

        # Reset clusters for each call and backstore for comparison
        self.clusters = {str(k): {"upper": [], "lower": []} for k in range(self.max_clusters)}

        # Assign each entity
        for k in range(0, self.data_length):
            v_clust = self.cluster_list[str(k)]     # Current entity optimal cluster
            T = {j: self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust],self.small])
                 for j in self.distance[str(k)] if
                 (self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust],self.small]) <= self.dist_threshold)
                 and (v_clust != j)}

            # Assign lower and upper approximations as needed
            if len(T.keys()) > 0:
                self.clusters[v_clust]["upper"].append(k)      # Assign entity to its nearest cluster upper approx.
                for cluster_name in T:
                    self.clusters[cluster_name]["upper"].append(k)  # Assign entity to upper approx of near cluster
            else:
                self.clusters[v_clust]["upper"].append(k)      # Assign entity to its nearest cluster upper approx.
                self.clusters[v_clust]["lower"].append(k)      # Assign entity to its nearest cluster lower approx.

            if self.debug_assign is True:
                print "Current Cluster", v_clust
                print "distance", self.distance[str(k)]
                print "T",T
        return

    def get_entity_centroid_distances(self):

        """
        Compute entity-cluster distances and find nearest cluster for each entity and assign

        uses self.data

        :return: self.distance : centroid-entity distance vectors for all candidate clusters
        """

        t1 = time.time()

        # Enumerate centroid distance vector for all entities and find nearest cluster and assign
        self.groups = {str(k): [] for k in range(self.max_clusters)}
        for k in range(0,self.data_length):
            self.distance[str(k)] = {str(j): np.linalg.norm([abs(self.data[val][k]-self.centroids[str(j)][val])
                                                             for val in self.feature_names])
                                     for j in range(self.max_clusters)}

            best_key = min(self.distance[str(k)].iteritems(), key=operator.itemgetter(1))[0]
            self.cluster_list[str(k)] = best_key
            self.groups[best_key].append(k)

        if self.debug_dist is True:
            print "Cluster List",self.cluster_list
            print "Distances",self.distance

        # Determine self.dist_threshold based on percentile all entity-cluster distances
        #curr_dists = list(itertools.chain([self.distance[h][g] for h in self.distance for g in self.distance[h]]))
        #self.dist_threshold = np.percentile(curr_dists,50)

        # if self.debug is True:
        #     print "Current Distances",curr_dists
            #print "Distance Threshold",self.dist_threshold

        #
        # self.all_keys = {str(key) : None for key in range(0,data_length)}	# Static all entity keys
        # curr_keys = {str(key) : None for key in range(0,data_length)}		# Place holder entity keys
        # self.total_entities = len(curr_keys.keys())
        #

        t2 = time.time()

        return

if __name__ == "__main__":

    """
    For class-level tests see /tests/rough_clustering_tests.py
    """
    from scipy.cluster.vq import kmeans2

    # Some tiny unit tests to be pushed to /tests/ as well later
    data = {"test1": [1.0,1.0,2.1],"test2": [2.0,2.01,2.3],"test3": [3.,3.,3.1]}
    datav = np.asarray([data["test1"],data["test2"],data["test3"]])
    print datav.T
    print "Data input",datav.shape
    [centroids, groups] = kmeans2(datav.T,2,iter=20)
    print centroids, groups
    clstr = RoughKMeans(data,2,0.75,0.25,1.1)
    clstr.get_rough_clusters()
    print "Final Rough k-means",clstr.cluster_list

