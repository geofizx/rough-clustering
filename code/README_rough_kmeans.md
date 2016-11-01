# Python Implementation of Rough K-means Clustering for float/int Feature Sets

###Description
    This algorithm clusters M-dimensional feature sets with N number of instances (entities) based on the Euclidean
    norm measure, D, between all entities and candidate cluster centroids.

    The basic objective is to divide entities into discernible (uniquely clustered) entities and
    indiscernible (shared) entities by assigning these entities to subsets. Clusters are based on the Euclidean
    distance as in conventional k-means clustering; however, rough k-means extends conventional k-means by making
    use of three properties of rough sets to compute optimal centroids of clusters (x_i) from the input feature set:

        Upper Approximation - A_sup(x_i) - Set of all entities in a cluster that may be shared with other clusters.

        Lower Approximation - A_sub(x_i) - Subset of Upper Approximation with entities unique to that cluster, i.e.,
                              discernible entities

        Boundary Region - A_sup(x_i) - A_sub(x_i) - Difference between Upper and Lower Approximation which contain strictly
        non-unique (shared) entities, i.e., indiscernible and entities belongs to two or more upper approximations

    In rough k-means, the entity membership of the Boundary Region (shared entities across clusters) of each cluster is
    determined based on a distance threshold, dt, as:

        For any entity c_i, let T be the set {clusters x_j for any j where || c_i - x_j || <= dt}

        if T != {0}, for all x_j in T, c_i will be assigned to their upper approximation
        else, x_i is assigned to the lower and upper approximation of its optimal cluster

    Furthermore, optimal cluster centroids take in to account both upper and lower approximations by the relation:

        centroid(x_i) = wght_lower * (A_sub(x_i))/|A_sub(x_i)| +
                        wght_upper * (A_sup(x_i) - A_sub(x_i))/|A_sup(x_i) - A_sub(x_i)|

        where wght_lower and wght_upper are the relative importance (wght_lower + wght_upper == 1) of lower versus
        upper approximations (default=0.75,0.25, respectively).

    The resulting optimal clusters can then be tuned based on three parameters above, upper_weight, lower_weight, and dt.
    If dt is chosen to be 1.0, then the rough k-means solution will match that of conventional k-means. As dt is increased,
    so too will the roughness of the resulting clusters (Boundary Regions grow in size).

####Input

    This algorithm takes as input a dictionary with <feature_name> : list pairs (float/int features)

####Options
    max_clusters - integer corresponding to number of clusters to return
    wght_lower (default=0.75)     - Relative weight of lower approximation for each rough cluster centroid
    wght_upper (default=0.25)     - Relative weight of upper approximation to each rough cluster centroid
    dist_threshold (default=1.25) - Threshold for clusters to be considered similar distances


####Optimized Clusters


####Usage

    /tests/rough_kmeans_tests.py - example usage and tests for known 2-class clustering problem
    /tests/rough_kmeans_iris.py - example usage and tests for known 3-class UCI Iris Data Set clustering problem

####Test Data Notes



####References

    Lingras & Peter, 2012, Applying Rough Set Concepts to Clustering, in G. Peters et al. (eds.), Rough Sets: Selected
    Methods and Applications in Management and Engineering, Advanced Information and Knowledge Processing,
    DOI 10.1007/978-1-4471-2760-4_2, Springer-Verlag London Limited.
