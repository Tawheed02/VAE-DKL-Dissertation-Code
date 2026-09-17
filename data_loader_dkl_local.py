import torch
from torch.utils.data import TensorDataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
# function used when running on gpu laptop

def get_supervised_loader(parquet_path, batch_size=4096, num_workers=0,
                           target_col='credit_spread', shuffle=True):

    exclude_cols = {'credit_spread', 'ytm', 'cusip', 'permno', 'permco', 'gvkey', 'date'}

    table = pq.read_table(parquet_path)
    schema_cols = table.schema.names
    feature_cols = [c for c in schema_cols if c.lower() not in exclude_cols]

    df = table.to_pandas()
    x_np = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in feature_cols])
    y_np = df[target_col].to_numpy(dtype=np.float32)

    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(y_np)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, len(feature_cols)
