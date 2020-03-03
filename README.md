# Cross-layer Optimization for High Speed Adders: A Pareto Driven Machine Learning Approach

## Data

The dataset is *data/label-22000.csv*. Each row contains a sequence representing prefix adder, two configuration parameters (max_delay and utilization) in back-end tools, and four values indicating area, power, delay and TNS.

*data/feature-1100.csv* combines the features for each instance.
Each row corresponds to an adder instance with the format as below:
  
\<INDEX\>,\<Sequence\>,\<SIZE\>,\<MFO\>,\<MPFO(32)\>,\<SPFO(32)\>

(MPFO refers to **max-path fanout**.)


