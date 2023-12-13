import numpy as np
import torch

def random_sampler(ptcl, num_samples)->torch.Tensor:
    input_dims = len(ptcl.shape)
    if isinstance(ptcl, np.ndarray):
        ptcl =  torch.Tensor(ptcl)
    if input_dims == 3:
        ptcl = torch.squeeze(ptcl,0)
    n_input_points = len(ptcl)
    n_repeats = int(np.floor(num_samples/n_input_points))
    if n_repeats > 0:
        idxs = torch.cat([torch.arange(0,n_input_points,dtype=int) for _ in range(n_repeats)])
    else:
        idxs = torch.zeros((0,),dtype=int)
    samples_to_gen = int(num_samples - n_repeats*n_input_points)
    random_idxs = torch.randperm(n_input_points)[:samples_to_gen]
    idxs=torch.cat((idxs,random_idxs))
    ptcl.to("cpu")
    return ptcl[idxs]

if __name__ == "__main__":
    import pyvista as pv
    from pyvista.plotting.plotter import Plotter
    ptcl = np.random.rand(1200,3)
    n_samples = 3000
    print(random_sampler(ptcl,n_samples).shape)
    n_samples = 400
    print(random_sampler(ptcl,n_samples).shape)
    plotter = Plotter()
    plotter.add_mesh(pv.PolyData(ptcl), color ="r",point_size=10)
    plotter.add_mesh(pv.PolyData(np.array(random_sampler(ptcl,1200))), color ="b")
    plotter.show()