# simple-factorgraph-example
A simple factor graph example with [gtsam](https://github.com/borglab/gtsam).

It solves a simple planar SLAM problem based on [an example in the gtsam library](https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py),
and plots the marginal distributions mapped from the manifold onto the translation plane.

I have made two implementations:
  - [batch_factorgraph_example.py](batch_factorgraph_example.py): 
    Batch procedure which optimizes one factor graph over all factors.
  - [incremental_factorgraph_example.py](incremental_factorgraph_example.py): 
    Incremental procedure which lets us add measurements incrementally and optimize using ISAM2.

## Dependencies
You can install all dependencies in [requirements.txt](requirements.txt) with pip:
```bash
pip install -r requirements.txt
```
