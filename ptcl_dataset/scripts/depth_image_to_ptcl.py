import numpy as np
import pyvista as pv
from depth_image_dataset.dataset import DepthImageDistanceFeaturesDataset
from ptcl_dataset.dataset import PtclDistanceFeaturesDataset
from tqdm import tqdm


class DepthImageToPtclConversor:
    def __init__(
        self,
        height,
        width,
        is_inverse,
        is_normalized,
        max_dist=None,
        void_value=None,
        n_folds=None,
    ):
        self.height = height
        self.width = width
        self.is_normalized = is_normalized
        self.is_inverse = is_inverse
        self.max_dist = max_dist
        self.void_value = void_value
        self.n_folds = n_folds
        if self.is_normalized or self.is_inverse:
            assert not self.max_dist is None
        self.theta_angles = np.reshape(
            np.hstack([np.linspace(0, np.pi * 2, self.width) for _ in range(height)]),
            -1,
        )
        self.delta_angles = np.reshape(
            np.repeat(np.linspace(np.deg2rad(15), np.deg2rad(-15), height), width), -1
        )

    def correct_image(self, image):
        if self.is_normalized:
            image *= self.max_dist
        if self.is_inverse:
            image = self.max_dist - image
        return image

    def get_valid_indices(self, image):
        if not self.void_value is None:
            return np.where(image != self.void_value)
        elif self.is_inverse and self.is_normalized:
            return np.where(image != 0)
        elif self.is_inverse and not self.is_normalized:
            return np.where(image != 0)
        elif not self.is_inverse and self.is_normalized:
            return np.where(image != 1)
        elif not self.is_inverse and not self.is_normalized:
            return np.where(image != self.max_dist)

    def __call__(self, image):
        if not self.n_folds is None:
            image = unfold_image(image, self.n_folds)
        image = np.reshape(image, -1)
        idxs_to_include = self.get_valid_indices(image)
        distances = self.correct_image(image)
        disttu = distances[idxs_to_include]  # disttu = distances to use
        thettu = self.theta_angles[idxs_to_include]  # thettu = thetas to use
        delttu = self.delta_angles[idxs_to_include]  # delttu = deltas to use
        horizontal_distances = disttu * np.cos(delttu)
        x = horizontal_distances * np.cos(thettu)
        y = horizontal_distances * np.sin(thettu)
        z = disttu * np.sin(delttu)
        return np.vstack([x, y, z])


def unfold_image(image, n_folds):
    image = np.squeeze(image)
    height, width = image.shape
    assert height % 2**n_folds == 0
    for n_stack in range(1, n_folds + 1):
        new_height = int(height / 2**n_stack)
        image = np.hstack((image[:new_height, :], image[new_height:, :]))
    return image


def main():
    # USER PARAM FOR TESTING
    image_dataset = DepthImageDistanceFeaturesDataset(
        name="depth_image_with_3_stackings_inveted_normalized",
        samples_to_generate=0,
        mode="read",
    )
    ptcl_dataset = PtclDistanceFeaturesDataset(
        name="converted_dataset",
        mode="write",
        identifiers={
            "max_distance": 30,
            "only_coords": True,
        },
    )
    ptcl_dataset.new_env("/home/lorenzo/gazebo_worlds/modified_playpen")
    images = image_dataset.get_loaded_inputs()
    labels = image_dataset.get_loaded_labels()
    conversor = DepthImageToPtclConversor(16,1024,True,True,30.0,0,3) 
    for image, label in tqdm(zip(images, labels),total=len(labels)):
        ptcl = conversor(image)
        ptcl_dataset.write_datapoint(ptcl,label)

if __name__ == "__main__":
    main()
