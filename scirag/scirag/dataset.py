from .config import datasets_path, DATASET
import pandas as pd

class SciRagDataSet:
    def __init__(self, dataset_name: str = DATASET):
        self.dataset_name = dataset_name
        self.dataset_path = datasets_path / dataset_name

    def load_dataset(self):
        return pd.read_parquet(self.dataset_path)
    
    
    