#!/usr/bin/env python2.7
# encoding: utf-8

"""
@description
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

@notes
Distance threshold option:
    self.dist_threshold = 1.25 by default (entity assigned to all centroids within 25% of the optimal cluster distance)
    if self.dist_threshold is set <=1.0 conventional kmeans clusters will be returned

    The larger self.dist_threshold the more rough (entity overlap) will exist across all k clusters returned

Lower and Upper Approximation Weight options:
    SUM(wght_lower,wght_upper) must equal 1.0, else it will be set to defaults on execution

    wght_lower=0.75 by default
    wght_upper=0.25 by default

    The larger wght_lower is relative to wght_upper the more important cluster lower approximations will be and v.v

@author Michael Tompkins
@copyright 2016
"""

# Externals
import warnings
import time
import operator
import numpy as np
from copy import deepcopy


class RoughKMeans:

    def __init__(self,input_data,max_clusters,wght_lower=0.75,wght_upper=0.25,threshold=1.25):

        # Rough clustering options
        self.max_clusters = max_clusters    # Number of clusters to return
        self.dist_threshold = threshold     # <=1.0 Threshold for centroids to be deemed indiscernible (1.0 == Kmeans)
        self.tolerance = 1.0e-02            # Tolerance for stopping iterative clustering
        self.wght_lower = wght_lower        # Relative weight of lower approximation for each rough cluster centroid
        self.wght_upper = wght_upper        # Relative weight of upper approximation to each rough cluster centroid

        # Enforce wght_lower + wght_upper == 1.0
        if self.wght_lower + self.wght_upper > 1.0:
            self.wght_lower = 0.75
            self.wght_lower = 0.25
            warnings.warn("Upper + Lower Weights must == 1.0, Setting Values to Default")

        # Rough clustering internal vars
        self.data = input_data
        self.data_array = None
        self.feature_names = input_data.keys()
        self.data_length = len(self.data[self.feature_names[0]])

        # Rough clustering external vars
        self.centroids = {}                 # Centroids for all returned clusters
        self.cluster_list = {}              # Internal listing of membership for all candidate clusters
        self.distance = {}                  # Entity-cluster distances for all candidate clusters
        self.clusters = None                # upper and lower approximation membership for all returned clusters

        # Overhead
        self.timing = True                  # Timing print statements flag
        self.debug = False                  # Debug flag for entire class print statements
        self.debug_assign = False           # Debug flag for assign_cluster_upper_lower_approximation method
        self.debug_dist = False             # Debug flag for get_entity_centroid_distances method
        self.debug_update = False           # Debug flag for update_centroids method
        self.small = 1.0e-04
        self.large = 1.0e+10

    def get_rough_clusters(self):

        """
        Run iterative clustering solver for rough k-means and return max_cluster rough clusters

        :return: self.centroids, self.assignments, self.upper_approximation, self.lower_approximation
        """

        # Transform data to nd-array for speed acceleration
        self.transform_data()

        # Get initial random entity clusters
        self.initialize_centroids()

        if self.dist_threshold <= 1.0:
            warnings.warn("Rough distance threshold set <= 1.0 and will produce conventional k-means solution")

        # Iterate until centroids convergence
        ct = 0
        stop_flag = False
        while stop_flag is False:

            t1 = time.time()
            # Back-store centroids
            prev_centroids = deepcopy(self.centroids)

            # Get entity-cluster distances
            self.get_entity_centroid_distances()

            # Compute upper and lower approximations
            self.assign_cluster_upper_lower_approximation()

            # Update centroids with upper and lower approximations
            self.update_centroids()

            # Determine if convergence reached
            stop_flag = self.get_centroid_convergence(prev_centroids)

            t2 = time.time()
            iter_time = t2-t1
            print "Clustering Iteration", ct, " in: ", iter_time," secs"
            ct += 1

        return

    def transform_data(self):

        """
        Convert input data dictionary to float nd-array for accelerated clustering speed

        :var self.data
        :return: self.data_array
        """

        t1 = time.time()

        tableau_lists = [self.data[key][:] for key in self.data]
        self.data_array = np.asfarray(tableau_lists).T

        if self.timing is True:
            t3 = time.time()
            print "transform_data Time",t3-t1
            print "shape",self.data_array.shape

    def initialize_centroids(self):

        """
        Randomly select [self.max_clusters] initial entities as centroids and assign to self.centroids

        :var self.max_clusters
        :var self.data
        :var self.data_array
        :var self.feature_names
        :return: self.centroids : current cluster centroids
        """

        t1 = time.time()

        # Select max cluster random entities from input and assign as initial cluster centroids
        candidates = np.random.permutation(self.data_length)[0:self.max_clusters]

        if self.debug is True:
            print "Candidates",candidates,self.feature_names,self.data

        #self.centroids = {str(k): {v: self.data[v][candidates[k]] for v in self.feature_names} for
        #                  k in range(self.max_clusters)}

        self.centroids = {str(k): self.data_array[candidates[k],:] for k in range(self.max_clusters)}

        if self.timing is True:
            t3 = time.time()
            print "initialize_centroids Time",t3-t1

        return

    def get_centroid_convergence(self,previous_centroids):

        """
        Convergence test. Determine if centroids have changed, if so, return False, else True

        :arg previous_centroids : back stored values for last iterate centroids
        :var self.centroids
        :var self.feature_names
        :var self.tolerance
        :return boolean : centroid_error <= self.tolerance (True) else (false)
        """

        t1 = time.time()

        # centroid_error = np.sum([[abs(self.centroids[k][val] - previous_centroids[k][val]) for k in self.centroids]
        #                           for val in self.feature_names])

        centroid_error = np.sum([np.linalg.norm(self.centroids[k] - previous_centroids[k]) for k in self.centroids])

        if self.timing is True:
            t3 = time.time()
            print "get_centroid_convergence Time",t3-t1, " with error:",centroid_error

        if self.debug is True:
            print "Centroid change", centroid_error

        if centroid_error <= self.tolerance:
            return True
        else:
            return False

    def update_centroids(self):

        """
        Update rough centroids for all candidate clusters given their upper/lower approximations set membership

        Cluster centroids updated/modified for three cases:
            if sets {lower approx} == {upper approx}, return conventional k-means cluster centroids
            elif set {lower approx] is empty and set {upper approx} is not empty, return upper-lower centroids
            else return weighted mean of lower approx centroids and upper-lower centroids

        :var self.data_array
        :var self.wght_lower
        :var self.wght_upper
        :var self.feature_names
        :var self.clusters
        :return: self.centroids : updated cluster centroids
        """

        t1 = time.time()

        for k in self.clusters:

            if len(self.clusters[k]["lower"]) == len(self.clusters[k]["upper"]):
                # Get lower approximation vectors
                #lower = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["lower"]])
                #self.centroids[str(k)] = {v: np.mean(lower[:,p]) for p, v in enumerate(self.feature_names)}
                lower = self.data_array[self.clusters[k]["lower"], :]
                self.centroids[str(k)] = np.mean(lower,axis=0)

            elif len(self.clusters[k]["lower"]) == 0 and len(self.clusters[k]["upper"]) != 0:
                # Get upper approximation vectors
                #upper = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["upper"]])
                #self.centroids[str(k)] = {v: np.mean(upper[:,p]) for p, v in enumerate(self.feature_names)}
                upper = self.data_array[self.clusters[k]["upper"], :]
                self.centroids[str(k)] = np.mean(upper,axis=0)

            else:
                # Get both upper and lower approximation vectors
                #upper = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["upper"]])
                #lower = np.asarray([[self.data[i][j] for i in self.feature_names] for j in self.clusters[k]["lower"]])
                # self.centroids[str(k)] = {v: self.wght_lower*np.mean(lower[:,p],axis=0) +
                #                         self.wght_upper*np.mean(upper[:,p],axis=0)
                #                         for p, v in enumerate(self.feature_names)}
                upper = self.data_array[self.clusters[k]["upper"], :]
                lower = self.data_array[self.clusters[k]["lower"], :]
                self.centroids[str(k)] = self.wght_lower*np.mean(lower,axis=0) + self.wght_upper*np.mean(upper,axis=0)

            if self.debug_update is True:
                print """###Cluster""", k, self.clusters[k]["lower"], self.clusters[k]["upper"]

        if self.timing is True:
            t3 = time.time()
            print "update_centroids Time", t3 - t1

        return

    def assign_cluster_upper_lower_approximation(self):

        """
        Compute entity-to-cluster optimal assignments + upper/lower approximations for all current clusters

        :var self.distance
        :var self.distance_threshold
        :var self.cluster_list
        :var self.max_clusters
        :var self.data_length
        :return: self.clusters[clusters]["upper"] : upper approximation for each candidate cluster
        :return: self.clusters[clusters]["lower"] : lower approximation for each candidate cluster
        """

        t1 = time.time()

        # Reset clusters for each method call
        self.clusters = {str(k): {"upper": [], "lower": []} for k in range(self.max_clusters)}

        # Assign each entity to cluster upper/lower approximations as appropriate
        for k in range(0, self.data_length):
            v_clust = self.cluster_list[str(k)]     # Current entity nearest cluster

            # Compile all clusters for each entity that are within self.threshold of best entity cluster
            T = {j: self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust],self.small])
                 for j in self.distance[str(k)] if
                 (self.distance[str(k)][j] / np.max([self.distance[str(k)][v_clust],self.small]) <= self.dist_threshold)
                 and (v_clust != j)}

            # Assign entity to lower and upper approximations of all clusters as needed
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

        if self.timing is True:
            t3 = time.time()
            print "assign_cluster_upper_lower_approximation Time", t3 - t1

        return

    def get_entity_centroid_distances(self):

        """
        Compute entity-cluster distances and find nearest cluster for each entity and assign for all entities

        :var self.data_array : numpy nd-array of all features for all entities
        :var self.centroids : numpy nd-array of all centroid features for all candidate clusters
        :var self.max_clusters
        :return: self.distance : centroid-entity distance vectors for all candidate clusters
        :return self.cluster_list : best fit cluster - entity assignment lists
        """

        t1 = time.time()

        # Enumerate centroid distance vector for all entities and find nearest cluster and assign
        # distance1 = {}
        # for k in range(0,self.data_length):
        #     distance1[str(k)] = {str(j): np.linalg.norm([abs(self.data[val][k]-self.centroids[str(j)][val])
        #                                                      for val in self.feature_names])
        #                              for j in range(self.max_clusters)}
        #
        #     best_key = min(distance1[str(k)].iteritems(), key=operator.itemgetter(1))[0]
        #     self.cluster_list[str(k)] = best_key
        # t2 = time.time()

        tmp = []
        for l in range(0,self.max_clusters):
            tmp.append(np.linalg.norm(self.data_array - np.asarray(self.centroids[str(l)]),axis=1))

        for k in range(0,self.data_length):
            self.distance[str(k)] = {str(j): tmp[j][k] for j in range(self.max_clusters)}
            best_key = min(self.distance[str(k)].iteritems(), key=operator.itemgetter(1))[0]
            self.cluster_list[str(k)] = best_key

        if self.debug_dist is True:
            print "Cluster List",self.cluster_list
            print "Distances",self.distance

        # Determine self.dist_threshold based on percentile all entity-cluster distances
        # curr_dists = list(itertools.chain([self.distance[h][g] for h in self.distance for g in self.distance[h]]))
        # self.dist_threshold = np.percentile(curr_dists,50)

        if self.timing is True:
            t3 = time.time()
            print "get_entity_centroid_distances Time",t3-t1

        return

if __name__ == "__main__":

    """
    For class-level tests see /tests/rough_kmeans_tests.py
    """

    # Class Unit test
    data = {"test1": [1.0,1.0,2.1],"test2": [2.0,2.01,2.3],"test3": [3.,3.,3.1]}
    clstr = RoughKMeans(data,2,0.75,0.25,1.1)
    clstr.get_rough_clusters()
    print "Final Rough k-means",clstr.cluster_list

