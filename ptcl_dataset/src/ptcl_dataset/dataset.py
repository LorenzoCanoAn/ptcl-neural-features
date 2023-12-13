from dataset_management.dataset import DatasetFileManagerToPytorchDataset
from dataset_management.dataset_io import DataFoldersManager
from ptcl_dataset.fps import farthest_point_sampler
from ptcl_dataset.random_sampling import random_sampler
from ptcl_dataset.knn_encoding import knn_encode
import math
import numpy as np
from tqdm import tqdm
import torch
import random


def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def T_to_xyzrpy(T):
    translation = T[:3, 3]
    x, y, z = translation
    rotation_matrix = T[:3, :3]
    pitch = -math.asin(rotation_matrix[2, 0])
    if math.cos(pitch) != 0:
        yaw = math.atan2(
            rotation_matrix[1, 0] / math.cos(pitch),
            rotation_matrix[0, 0] / math.cos(pitch),
        )
    else:
        yaw = 0
    roll = math.atan2(
        rotation_matrix[2, 1] / math.cos(pitch), rotation_matrix[2, 2] / math.cos(pitch)
    )
    return x, y, z, roll, pitch, yaw


def xyzrpyw_to_T(x, y, z, roll, pitch, yaw):
    translation_matrix = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0, 0],
            [sin_yaw, cos_yaw, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    rotation_pitch = np.array(
        [
            [1, 0, 0, 0],
            [0, cos_pitch, -sin_pitch, 0],
            [0, sin_pitch, cos_pitch, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rotation_roll = np.array(
        [
            [cos_roll, 0, sin_roll, 0],
            [0, 1, 0, 0],
            [-sin_roll, 0, cos_roll, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_matrix = np.dot(np.dot(rotation_yaw, rotation_pitch), rotation_roll)
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix


def gen_label(transform1, transform2):
    x1, y1, z1, r1, p1, yw1 = np.reshape(np.array(transform1), -1)
    x2, y2, z2, r2, p2, yw2 = np.reshape(np.array(transform2), -1)
    T1 = xyzrpyw_to_T(x1, y1, z1, r1, p1, yw1)
    T2 = xyzrpyw_to_T(x2, y2, z2, r2, p2, yw2)
    T12 = np.dot(np.linalg.inv(T1), T2)
    x, y, z, roll, pitch, yaw = T_to_xyzrpy(T12)
    qx, qy, qz, qw = euler_to_quaternion(yaw, pitch, roll)
    return x, y, z, qx, qy, qz, qw


class PtclDistanceFeaturesDataset(DatasetFileManagerToPytorchDataset):
    required_identifiers = ["max_distance", "only_coords"]

    def __init__(
        self,
        datafolders_manager=DataFoldersManager.get_current_instance("ptcl_datasets"),
        name=None,
        mode="read",
        identifiers=dict(),
        unwanted_characteristics=dict(),
        samples_to_load: int = None,
        samples_to_generate: int = 100,
        prob_of_same_pose_sample: float = 0.5,
        ptcl_size: int = 2000,
        pre_sample: bool = True,
        n_knn_encoding: int = 5,
        z_cutoff=None,
        sampling_function=random_sampler,
    ):
        super().__init__(
            datafolders_manager,
            name,
            mode,
            identifiers,
            unwanted_characteristics,
            samples_to_load,
            samples_to_generate=samples_to_generate,
            prob_of_same_pose_sample=prob_of_same_pose_sample,
            ptcl_size=ptcl_size,
            pre_sample=pre_sample,
            n_knn_encoding=n_knn_encoding,
            z_cutoff=z_cutoff,
            sampling_function=sampling_function,
        )

    def process_raw_inputs(self, samples_to_generate=None):
        if not self.z_cutoff is None:
            for idx, ptcl in tqdm(
                enumerate(self._loaded_inputs),
                desc="z_cuttof",
                total=(len(self._loaded_inputs)),
            ):
                ptcl = ptcl.T
                self._loaded_inputs[idx] = ptcl[ptcl[:, 2] > self.z_cutoff, :]
        if not samples_to_generate is None:
            self.samples_to_generate = int(samples_to_generate)
        # In the same dataset there can be more than one env -> it is necessary to create the labels env per env
        n_datapoints_per_datafolder = (
            [
                datafolder.n_files
                for datafolder in self.input_manager.selected_datafolders
            ]
            if self.samples_to_load is None
            else [self.samples_to_load for _ in self.input_manager.selected_datafolders]
        )
        total_datapoints = len(self._loaded_labels)
        if total_datapoints == self.samples_to_generate:
            samples_to_generate_per_datafolder = n_datapoints_per_datafolder
        else:
            samples_to_generate_per_datafolder = []
            for n_datapoints in n_datapoints_per_datafolder:
                samples_to_generate_per_datafolder.append(
                    int(n_datapoints / total_datapoints * self.samples_to_generate)
                )
            if sum(samples_to_generate_per_datafolder) != self.samples_to_generate:
                samples_to_generate_per_datafolder[
                    -1
                ] += self.samples_to_generate - sum(samples_to_generate_per_datafolder)
        self.start_idx = 0
        self._inputs = [None for _ in range(self.samples_to_generate)]
        self._labels = [None for _ in range(self.samples_to_generate)]
        for n_datafolder, n_samples_to_generate in tqdm(
            enumerate(samples_to_generate_per_datafolder), desc="Env progression"
        ):
            self.start_idx_for_loaded = sum(n_datapoints_per_datafolder[:n_datafolder])
            self.end_idx_for_loaded = sum(
                n_datapoints_per_datafolder[: n_datafolder + 1]
            )

            for n_sample in tqdm(range(n_samples_to_generate), desc="Gen samples"):
                self.calc_sample(n_sample)
            self.start_idx += n_samples_to_generate

    def calc_sample(self, n_sample):
        while True:
            idx1 = random.randint(
                self.start_idx_for_loaded, self.end_idx_for_loaded - 1
            )
            idx2 = random.randint(
                self.start_idx_for_loaded, self.end_idx_for_loaded - 1
            )
            label = gen_label(self._loaded_labels[idx1], self._loaded_labels[idx2])
            x, y, z, qx, qy, qz, qw = label
            if np.linalg.norm(np.array((x, y, z))) < 1.5:
                break
        ptcl1 = self._loaded_inputs[idx1]
        ptcl2 = self._loaded_inputs[idx2]
        if ptcl1.shape[0] == 3:
            ptcl1 = ptcl1.T
        if ptcl2.shape[0] == 3:
            ptcl2 = ptcl2.T
        if self.pre_sample:
            ptcl1 = self.sampling_function(ptcl1, self.ptcl_size)
            ptcl2 = self.sampling_function(ptcl2, self.ptcl_size)
        if not self.n_knn_encoding is None:
            ptcl1 = knn_encode(ptcl1, self.n_knn_encoding)
            ptcl2 = knn_encode(ptcl2, self.n_knn_encoding)
        ptcl1 = ptcl1.to("cpu")
        ptcl2 = ptcl2.to("cpu")
        self._inputs[self.start_idx + n_sample] = (ptcl1, ptcl2)
        self._labels[self.start_idx + n_sample] = torch.Tensor(
            (x, y, z, qx, qy, qz, qw)
        )

    def import_args(
        self,
        samples_to_generate,
        prob_of_same_pose_sample,
        ptcl_size,
        pre_sample,
        n_knn_encoding,
        z_cutoff,
        sampling_function,
    ):
        self.samples_to_generate = samples_to_generate
        self.prob_of_same_pose_sample = prob_of_same_pose_sample
        self.ptcl_size = ptcl_size
        self.pre_sample = pre_sample
        self.n_knn_encoding = n_knn_encoding
        self.z_cutoff = z_cutoff
        self.sampling_function = sampling_function
        if not self.n_knn_encoding is None:
            assert isinstance(self.n_knn_encoding, int)
            self.pre_sample = True

    def __getitem__(self, idx):
        "This has to overwritten to adjust the ptcl size"
        inputs, labels = self._inputs[idx], self._labels[idx]
        if not self.pre_sample:
            ptc1 = self.sampling_function(inputs[0], self.ptcl_size)
            ptc2 = self.sampling_function(inputs[1], self.ptcl_size)
        else:
            ptc1 = inputs[0]
            ptc2 = inputs[1]
        return (ptc1, ptc2), labels
