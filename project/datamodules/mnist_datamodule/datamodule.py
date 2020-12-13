from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """
    This is example of datamodule for MNIST dataset.

    The path to datamodule should be specified in your run config. (run_configs.yaml)
    The 'hparams' dict contains your hparams specified in run config. (run_configs.yaml)
    """

    def __init__(self, data_dir: str, hparams: dict):
        super().__init__()

        self.data_dir = data_dir + "/MNIST"  # data_dir is specified in project_config.yaml

        self.batch_size = hparams.get("batch_size") or 64
        self.train_val_test_split = hparams.get("train_val_test_split") or [55_000, 5_000, 10_000]
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

        self.transforms = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.data_dir, train=False, transform=self.transforms)
        dataset = ConcatDataset(datasets=[trainset, testset])

        self.data_train, self.data_val, self.data_test = random_split(dataset, self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
