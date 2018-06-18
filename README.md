# nonstick
This library contains a Python implementation of the probabilistic analysis of molecular motifs (PAMM) algorithm. PAMM provides a method of identifying reoccurring structural patterns in a molecular dynamics (MD) simulation. Our implementation includes the following steps:

1. convert raw atomic position data to local structure features
2. create a grid for density estimation with farthest point sampling
3. estimate the probability density in different areas of phase space with kernel density estimation
4. identify clusters in phase space with the quick shift mode seeking algorithm
5. build a gaussian mixture model to describe preidentified clusters

We suggest that you use this software in conjunction with the [PLUMED 2 software plugin](https://plumed.github.io/doc-master/user-doc/html/index.html), which can [incorporate the GMM output from PAMM into an enhanced sampling simulation](https://plumed.github.io/doc-master/user-doc/html/_p_a_m_m.html) in your MD engine of choice. You can also use the collective variable functionality in PLUMED to efficiently calculate local structural features (step 1 above) if you don't want to define these features yourself in Python.
