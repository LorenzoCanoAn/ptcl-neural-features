import pyvista as pv
from pyvista.plotting.plotter import Plotter
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    PATH_TO_FOLDER = "/home/lorenzo/.datasets/ptcl_datasets/0"
    filenames = os.listdir(PATH_TO_FOLDER)
    for filename in filenames:
        path_to_file = os.path.join(PATH_TO_FOLDER, filename)
        data = np.load(path_to_file)
        ptcl = data["input_data"].T
        ptcl_no_floor = ptcl[ptcl[:,2]>-0.6,:]
        ptcl_floor = ptcl[ptcl[:,2]<=-0.6,:]
        ptcl = ptcl_floor
        dist_to_z_axis = np.linalg.norm(ptcl[:,:2],ord=2,axis=1)
        #plotter = Plotter()
        #plotter.add_mesh(pv.PolyData(ptcl),scalars=[dist_to_z_axis]*ptcl[:,2])
        #plotter.show()
        plt.scatter(dist_to_z_axis, ptcl[:,2])
        plt.show()
        #plotter = Plotter()
        #plotter.add_mesh(pv.PolyData(ptcl_floor),scalars=ptcl_floor[:,2])
        #plotter.show()
        break
if __name__ == "__main__":
    main()
