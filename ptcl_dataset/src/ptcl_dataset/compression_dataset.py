from dataset_management.dataset import DatasetFileManagerToPytorchDataset
from dataset_management.dataset_io import DataFoldersManager
from ptcl_dataset.random_sampling import random_sampler
from ptcl_dataset.knn_encoding import knn_encode
import math
import numpy as np
from tqdm import tqdm
import torch
import random


class PTCL_dataset(DatasetFileManagerToPytorchDataset):
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
            sampling_function=sampling_function,
        )

    def process_raw_inputs(self,samples_to_generate=None):
        if samples_to_generate is None:
            samples_to_generate = len(self._loaded_inputs)
        self._inputs = [None for _ in range(samples_to_generate)] 
        self._labels = [torch.zeros([1]) for _ in range(samples_to_generate)] 
        for i in tqdm(range(samples_to_generate),desc="Processing dataset",total=samples_to_generate):
            self.calc_sample(i)
            
    def calc_sample(self, n_sample):
        ptcl1 = self._loaded_inputs[n_sample]
        if ptcl1.shape[0] == 3:
            ptcl1 = ptcl1.T
        if self.pre_sample:
            ptcl1 = self.sampling_function(ptcl1, self.ptcl_size)
        self._inputs[n_sample] = ptcl1

    def import_args(
        self,
        samples_to_generate,
        prob_of_same_pose_sample,
        ptcl_size,
        pre_sample,
        n_knn_encoding,
        sampling_function,
    ):
        self.samples_to_generate = samples_to_generate
        self.prob_of_same_pose_sample = prob_of_same_pose_sample
        self.ptcl_size = ptcl_size
        self.pre_sample = pre_sample
        self.n_knn_encoding = n_knn_encoding
        self.sampling_function = sampling_function
        if not self.n_knn_encoding is None:
            assert isinstance(self.n_knn_encoding, int)
            self.pre_sample = True

    def __getitem__(self, idx):
        "This has to overwritten to adjust the ptcl size"
        inputs, labels = self._inputs[idx], self._labels[idx]
        if not self.pre_sample:
            ptc1 = self.sampling_function(inputs, self.ptcl_size)
        else:
            ptc1 = inputs
        return ptc1, labels
