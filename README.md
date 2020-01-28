## The Next Best Question

The source code for TNBQ presented in *The Next Best Question: A Lazy-Anytime Framework for Adaptive Feature Acquisition* 

**Instructions:**
- Download the data sets from the links in each relevant folder.
- Run `python main.py` using the following flags
  - `--dataset` - the dataset name, 
  - `--method` - the method, out of 4 options `TNBQ`, `weighted`, `radius`, `global`
  - `--parameters` - the chosen method's parameters. 
    - `TNBQ` - K
    - `weighted` - K min_scale max_scale
    - `raiuds` - threshold min_neighborhood_size max_neighborhood_size
  - `--seed` - set the seed, optional
- E.g. `python main.py --dataset statlog --method weighted --parameters 3 0.5 1 --seed 10`
- The run will produce two files - the graph as a *.png* file and its details as *.txt* file in the *results* folder
