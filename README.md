## Fast Mean Shift (Dual Tree) Implementation

### Dual-Tree Mean Shift
Dual-Tree Mean Shift is an acceleration technique for computing the mean-shift algorithm. The idea is to use a kd-tree to represent both our query and reference sets, and use geometric properties of the partitioning of data to prune the tree. The technique was proposed by Wang, Lee, Gray, and Rehg in a paper at Georgia Tech in 2007.

### Resources
Link to the main paper is here: http://proceedings.mlr.press/v2/wang07d/wang07d.pdf <br>
Link to resource on dual-tree algorithms is here: https://arxiv.org/pdf/1304.4327.pdf <br>
Link to original paper on dual-tree search is here: http://proceedings.mlr.press/v28/curtin13.pdf 
