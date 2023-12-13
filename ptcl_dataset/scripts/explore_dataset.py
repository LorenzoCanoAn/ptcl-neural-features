from ptcl_dataset.dataset import PtclDistanceFeaturesDataset
import pyvista as pv
from pyvista.plotting.plotter import Plotter
import numpy as np

PARAMETERS = {
    "size_of_pcl": 3000,
    "loaded_samples": 1000,
    "generated_samples": 10,
}


def rotate_pointcloud_z(pointcloud, angle):
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotated_ptcl = np.dot(rotation_matrix, pointcloud.T)
    return rotated_ptcl.T


dataset = PtclDistanceFeaturesDataset(
    name="converted_dataset",
    mode="read",
    samples_to_generate=PARAMETERS["generated_samples"],
    ptcl_size=PARAMETERS["size_of_pcl"],
    samples_to_load=PARAMETERS["loaded_samples"],
    n_knn_encoding=None,
)


for i in range(10):
    plotter = Plotter()
    (ptcl1, ptcl2), label = dataset[i]
    x = label[0]
    y = label[1]
    yaw = np.rad2deg(label[2])
    rotated_ptcl2 = rotate_pointcloud_z(ptcl2, np.deg2rad(-yaw))
    print(f"x: {x.item()} y: {y.item()} yaw: {yaw.item()}")
    plotter.add_mesh(pv.PolyData(np.array(ptcl1)), color="b",point_size=15)
    plotter.add_mesh(pv.PolyData(np.array(ptcl2)), color="r",point_size=10)
    plotter.add_mesh(pv.PolyData(np.array(rotated_ptcl2)), color="g",point_size=5)
    plotter.add_axes_at_origin()
    plotter.show()
