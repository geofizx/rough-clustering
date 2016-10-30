# Python Implementation of Rough Set Clustering for Integer Feature Sets

###Description
    This algorithm clusters multi-dimensional feature sets with N number of instances (entities) based on an
    absolute integer-distance measure, D, between all entities (sum of all absolute feature differences between any 2 entities).

    The basic objective is to divide entities into discernible (uniquely clustered) entities and
    indiscernible (shared) entities by assigning these entities to subsets. Clusters are based on absolute distance D
    and not the Euclidean distance as in k-means.

    It also makes use of three properties of rough sets to enumerate these clusters (x_i) from the input feature set for entities:

        Upper Approximation - A_sup(x_i) - Set of all entities in a cluster that may be shared with other clusters.

        Lower Approximation - A_sub(x_i) - Subset of Upper Approximation with entities unique to that cluster, i.e., discernible entities

        Boundary Region - A_sup(x_i) - A_sub(x_i) - Difference between Upper and Lower Approximation which contain strictly
        non-unique (shared) entities, i.e., indiscernible and entities belongs to two or more upper approximations

    This code is an implementation of rough clustering as outlined by Voges et al., 2002

####Input

    This algorithm takes as input a dictionary with <feature_name> : list pairs (integer features)

####Options
    max_clusters - integer corresponding to number of clusters to return
    objective (default="ratio") - return max_clusters at distance D that maximizes this property of clusters
    max_d - Maximum inter-entity distance to consider before stopping further clustering

    if max_d is not specified, then algorithm determines max_d based on inter-entity distance (25th percentile)

####Optimized Clusters
    The algorithm determines the optimal inter-entity distance D for final clustering based on option 'objective' which maximizes :
    "lower" : sum of lower approximations - maximum entity uniqueness across all clusters at distance D
    "coverage" : total # of entites covered by all clusters - maximum number of entities across all clusters at distance D
    "ratio" : ratio of lower/coverage (default) - maximum ratio of unique entities to total entities across all clusters at distance D
    "all" : return clusters at every distance D from [0 - self.total_entities]

####Usage

    /tests/rough_clustering_tests.py - example usage and tests for known 2-class clustering problem in UCI Statlog Data
    set for credit risk

####Test Data Notes

    In these tests, resulting cluster mean and std deviations from centroids are compared to kmeans and true class
    statistics in the resulting graph

    Rough clusters do not represent unique entities, but do better represent feature statistics compared to k-means.

####References

    Voges, Pope & Brown, 2002, "Cluster Analysis of Marketing Data Examining On-line Shopping Orientation:
    A Comparison of k-means and Rough Clustering Approaches"
