import numpy as np
import torch

def knn_encode(ptcl, nknn, device="cuda"):
    """This funcion takes a ptcl of shape [N,3], and outputs a tensor of shape [N, 3 + nknn*3],
    where the extra columns are the vectors from the point at that row, to its n closest neighbors
    """
    if isinstance(ptcl, np.ndarray):
        ptcl = torch.Tensor(ptcl)
    ptcl = ptcl.to(device)
    if ptcl.shape[1] != 3:
        raise ValueError("Input tensor must have shape (N, 3)")

    # Compute the pairwise Euclidean distances
    difference_matrix = ptcl[:, None, :] - ptcl
    distance_matrix = torch.norm(difference_matrix,dim=2,p=2)
    mins = torch.topk(torch.negative(distance_matrix), nknn+1, dim=1).indices[:,1:]
    encoding = torch.vstack([torch.reshape(difference_matrix[i, ids,:],(-1,)) for i, ids in enumerate(mins)])
    return torch.hstack((ptcl, encoding))