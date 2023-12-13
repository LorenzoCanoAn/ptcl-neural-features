import numpy as np
#from dgl.geometry import farthest_point_sampler as _fps
import torch

def farthest_point_sampler(ptcl, num_samples)->torch.Tensor:
    return 
    input_dims = len(ptcl.shape)
    if isinstance(ptcl, np.ndarray):
        ptcl =  torch.Tensor(ptcl)
    if input_dims == 2:
        ptcl = torch.unsqueeze(ptcl, 0)
    ptcl.to("cpu")
    idxs = _fps(ptcl,num_samples) 
    sampled_ptcl = ptcl[0, idxs, :]
    if input_dims == 2:
        sampled_ptcl = sampled_ptcl[0]
    return sampled_ptcl