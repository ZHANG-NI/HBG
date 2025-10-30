# Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems

Our paper *“[Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](https://arxiv.org/abs/2510.04792)”* has been accepted to NeurIPS 2025.

The proposed HBG framework is compatible with existing GFlowNet-based solvers, including [GFACS](https://github.com/ai4co/gfacs/tree/main?tab=readme-ov-file) and [AGFN](https://github.com/ZHANG-NI/AGFN ). All datasets used in our experiments are derived from (https://github.com/ZHANG-NI/AGFN/tree/main/data).

#### Training

```bash
python train_tb+db.py $N -p "path_to_checkpoint"
```

$N is the nodes of the instances

#### Testing

```bash
python train_tb+db.py $N -p "path_to_checkpoint"
```

$N is the nodes of the instances

### **Reference**

If you find this codebase useful, please consider citing the paper:

```
@inproceedings{zhang2025hybrid,
  title={Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems},
  author={Zhang, Ni and Cao, Zhiguang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```






