from ptcl_neural_features.find_lr import LRFinder
from ptcl_neural_features.networks import PCTFCDiffDistEstimator
from ptcl_dataset.dataset import PtclDistanceFeaturesDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
PARAMETERS = {
    "size_of_pcl": 3000,
    "point_size": 3,
    "embedding_size": 128,
    "n_heads": 16,
    "n_layers": 8,
    "hidden_dim": 2024,
    "feature_size": 256,
    "output_size": 2,
    "dropout":0.1,
    "batch_size": 8,
    "loaded_samples": None,
    "generated_samples": 25000,
    "n_epochs":20,
}

dataset = PtclDistanceFeaturesDataset(
    name="converted_dataset",
    mode="read",
    samples_to_generate=PARAMETERS["generated_samples"],
    ptcl_size=PARAMETERS["size_of_pcl"],
    samples_to_load=PARAMETERS["loaded_samples"],
)
dataloader = DataLoader(dataset, batch_size=PARAMETERS["batch_size"])
model = PCTFCDiffDistEstimator(
    PARAMETERS["size_of_pcl"],
    PARAMETERS["point_size"],
    PARAMETERS["embedding_size"],
    PARAMETERS["n_heads"],
    PARAMETERS["n_layers"],
    PARAMETERS["hidden_dim"],
    PARAMETERS["feature_size"],
    PARAMETERS["output_size"],
    PARAMETERS["dropout"],
)
criterion = MSELoss(2)
optimizer = Adam(model.parameters(),1e-15)
lrfinder = LRFinder(model,optimizer,criterion,"cuda")
lrfinder.range_test(dataloader, start_lr=1e-15,step_mode="exp")
lrfinder.plot()
