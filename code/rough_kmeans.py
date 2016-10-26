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

    def __init__(self,input_data,max_clusters,wght_lower=0.75,wght_upper=0.25):

        # Clustering vars
        self.data = input_data
        self.feature_names = input_data.keys()
        self.data_length = len(self.data[self.feature_names[0]])
        self.centroids = {}
        self.cluster_list = {}
        self.distance = {}

        # Rough sets vars
        self.all_keys = {}
        self.clusters = []
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
        self.dist_threshold = None          # Threshold for clusters to be considered similar distances

        # Overhead
        self.debug = True
        self.small = 1.0e-10
        self.large = 1.0e+10
        self.centroid_error = self.large    # Initial error for early stopping
        self.tolerance = 1.0e-03  # Tolerance for stopping iterative clustering

    def get_rough_clusters(self):

        """
        Run iterative clustering solver for rough k-means and return max_cluster rough clusters
        :return: self. centroids, self.assignments, self.upper_approximation, self.lower_approximation
        """

        # Get initial random entity clusters
        self.initialize_centroids()

        while self.centroid_error > self.tolerance:

            # Back-store centroids
            prev_centroids = deepcopy(self.centroids)

            # Get entity-cluster distances
            self.get_entity_centroid_distances()

            # Compute upper and lower approximations
            self.get_cluster_upper_lower_approximation()

            # Update centroids with upper and lower approximations
            self.update_centroids()

            # Recompute centroid error
            self.get_centroid_convergence(prev_centroids)

            if self.debug is True:
                print "Centroid change", self.centroid_error

        return

    def get_centroid_convergence(self,previous_centroids):

        self.centroid_error = np.sum([[abs(self.centroids[k][val] - previous_centroids[k][val]) for k in self.centroids]
                                   for val in self.feature_names])

        return

    def initialize_centroids(self):

        """
        Randomly select [self.max_clusters] initial entities as centroids and assign to self.centroids
        :return: self.centroids : current cluster centroids
        """

        # Select max cluster random entities from input and assign as initial cluster centroids
        candidates = np.random.permutation(self.data_length)[0:self.max_clusters]

        if self.debug is True:
            print "Candidates",candidates,self.feature_names,self.data

        self.centroids = {k: {v : self.data[v][candidates[k]] for v in self.feature_names} for
                          k in range(self.max_clusters)}

        return

    def update_centroids(self):

        """
        Update modified rough centroids for all candidate clusters given their upper/lower approximations membership

        uses self.data
        uses wght_lower, wght_upper

        :return: self.centroids : updated cluster centroids
        """
        pass

        return

    def get_cluster_upper_lower_approximation(self):


        pass

        return

    def get_entity_centroid_distances(self):

        """
        Compute modified rough entity-centroid distance for all cluster centroids given their upper/lower approximations

        uses self.data
        :return: self.distance : centroid-entity distance vectors for all candidate clusters
        """

        t1 = time.time()

        # Enumerate centroid distance vector for all entities and find nearest cluster
        for k in range(0,self.data_length):
            self.distance[str(k)] = {str(j): np.linalg.norm([abs(self.data[val][k]-self.centroids[j][val])
                                                             for val in self.feature_names])
                                     for j in range(self.max_clusters)}

            self.cluster_list[str(k)] = np.argmin(list(itertools.chain([self.distance[str(k)][j]
                                                                        for j in self.distance[str(k)]])))

            if self.debug is True:
                print "Cluster List",self.cluster_list[str(k)]
                print "Distances",self.distance[str(k)]

        # Determine self.dist_threshold based on percentile all entity-cluster distances
        curr_dists = list(itertools.chain([self.distance[h][g] for h in self.distance for g in self.distance[h]]))
        self.dist_threshold = int(np.percentile(curr_dists,25))

        if self.debug is True:
            print "Current Distances",curr_dists
            print "Distance Threshold",self.dist_threshold

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

    # Some tiny unit tests to be pushed to /tests/ as well later
    data = {"test1":[0.1,2.0,3.0],"test2":[0.5,3.1,0.1],"test3":[2.1,2.3,3.1]}
    clstr = RoughKMeans(data,2)
    clstr.get_rough_clusters()

