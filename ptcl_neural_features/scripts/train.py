from ptcl_dataset.dataset import PtclDistanceFeaturesDataset
from ptcl_neural_features.networks import DistanceEstimatorCat
from pct_compressor.PCT_PCC import get_model as PCT_PCC
from pct_compressor.PCT_PCC import get_loss
from torch.utils.data import DataLoader
from time import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import pyvista as pv
from pyvista.plotting.plotter import Plotter 
# torch.set_float32_matmul_precision("high")

class MeanAngularErrorLoss(nn.Module):
    def forward(self, predicted_angles, target_angles):
        """
        Compute the mean angular error loss between predicted and target angles.

        Parameters:
        - predicted_angles (torch.Tensor): Predicted angles (output from your model).
        - target_angles (torch.Tensor): Target angles.

        Returns:
        - torch.Tensor: Mean angular error loss.
        """
        # Ensure angles are in the range [0, 2*pi)
        predicted_angles = torch.fmod(predicted_angles, 2 * np.pi)
        target_angles = torch.fmod(target_angles, 2 * np.pi)

        # Compute angular difference
        angular_diff = torch.abs(
            torch.atan2(
                torch.sin(predicted_angles - target_angles),
                torch.cos(predicted_angles - target_angles),
            )
        )

        # Compute mean angular error
        mean_angular_error = torch.mean(angular_diff)

        return mean_angular_error


# Setup loging
logdir = "/home/lorenzo/.tensorboard"
os.popen(f"rm -rf {logdir}/**")
writer = SummaryWriter(log_dir=logdir)
# Load dataset:
PARAMETERS = {
    "n_epochs": 200,
    "batch_size": 16,
    "size_of_ptcl": 1024,
    "point_size": 3,
    "feature_size": 256,
    "output_size": 2,
    "dropout": 0.3,
    "samples_to_gen":1000,
    "samples_to_load":None,
    "compression_learning_rate": 1e-3,
    "distance_learning_rate": 1e-4,
}
############################################
# SETUP MODELS
############################################
compressor_model = PCT_PCC(bottleneck_size=PARAMETERS["feature_size"], recon_points=PARAMETERS["size_of_ptcl"])
compressor_model = compressor_model.to("cuda")
odom_model = DistanceEstimatorCat(feature_size=PARAMETERS["feature_size"])
odom_model = odom_model.to("cuda")
############################################
# SETUP CRITERIONS
############################################
compression_criterion = get_loss(lam=1000)
odom_criterion = nn.MSELoss(2)
############################################
# SETUP OPTIMIZER
############################################
compression_optimizer = torch.optim.Adam(
    compressor_model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.StepLR(compression_optimizer, step_size=20, gamma=0.5)
odom_optimizer = torch.optim.Adam(odom_model.parameters(),lr=PARAMETERS["distance_learning_rate"])
############################################
# SETUP DATASET
############################################
dataset = PtclDistanceFeaturesDataset(
    name="converted_dataset",
    mode="read",
    samples_to_generate=PARAMETERS["samples_to_gen"],
    ptcl_size=PARAMETERS["size_of_ptcl"],
    samples_to_load=PARAMETERS["samples_to_load"],
    pre_sample=True,
    n_knn_encoding=None,
    z_cutoff = -0.6,  
)
train_length = int(0.9 * len(dataset))
test_length = int(len(dataset) - train_length)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_length, test_length]
)
############################################
# SAVE DIR
############################################
save_dir = f"/home/lorenzo/models/hey"
print(f"Save dir: {save_dir}")
os.makedirs(save_dir, exist_ok=True)
############################################
# TRAIN LOOP
############################################
counter = 0
compression_loss_achieved = False
for n_epoch in tqdm(
    range(PARAMETERS["n_epochs"]),
    desc="Epoch",
    leave=False,
):
    train_dataloader = DataLoader(train_dataset, PARAMETERS["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, 2, shuffle=True)
    n_iters = len(train_dataloader)
    compressor_model.train()
    compressor_model.to("cuda")
    for n_iter_in_epoch, sample in tqdm(
        enumerate(train_dataloader),
        leave=False,
        total=len(train_dataloader),
    ):
        # Get data
        (ptcl1, ptcl2), labels = sample
        ptcl1 = ptcl1.to("cuda")/30
        ptcl2 = ptcl2.to("cuda")/30
        labels = labels.to("cuda")
        compression_optimizer.zero_grad()
        # COMPRESSION PASS
        bppx, coor_reconx, cdx, compressed_x = compressor_model(ptcl1)
        bppy, coor_recony, cdy, compressed_y = compressor_model(ptcl2)
        plotter = Plotter(off_screen=True)
        plotter.add_mesh(pv.PolyData(ptcl1[0].detach().cpu().numpy()*5),color="b")
        plotter.add_mesh(pv.PolyData(coor_reconx[0].detach().cpu().numpy()*5),color="r")
        plotter.add_axes_at_origin()
        plotter.show(screenshot=f"/home/lorenzo/screenshot/{counter}.png")
        compression_lossx = compression_criterion(bppx, cdx)
        compression_lossy = compression_criterion(bppy, cdy)
        compression_loss = compression_lossx + compression_lossy 
        compression_loss.backward()
        compression_optimizer.step()
        writer.add_scalars(
            "compression_loss",
            {"x": compression_lossx, "y": compression_lossy},
            counter,
        )
        counter += 1
    scheduler.step()
    torch.save(
        compressor_model.to("cpu").state_dict(), os.path.join(save_dir, "compressor.torch")
    )
    torch.save(
        odom_model.to("cpu").state_dict(), os.path.join(save_dir, "odom.torch")
    )
