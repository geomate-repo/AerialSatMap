# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train SatMap with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/satmap/satmap_tiny_r50_24e.py 4
```

Eval SatMap with 1 GPUs
```
./tools/dist_test_map.sh ./projects/configs/satmap/satmap_tiny_r50_24e.py ./path/to/ckpts.pth 1
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/SatMap/tools/maptr`