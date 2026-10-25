import pandas as pd
from torch.utils.data import Dataset

class DataSource:
    def __init__(self):
        pass

    def get_tags(self) -> dict:
        raise NotImplementedError

    def get_torch_dataset(self) -> Dataset:
       raise NotImplementedError        

    def get_pandas_dataset(self) -> pd.DataFrame:
        raise NotImplementedError