from dataclasses import dataclass
import torch

@dataclass
class TrackOnCfg:
    input_size: list = (384, 512)
    N: int = 384
    T: int = 18
    stride: int = 4
    transformer_embedding_dim: int = 256
    cnn_corr: bool = False
    linear_visibility: bool = False

    num_layers: int = 3
    num_layers_offset_head: int = 3

    num_layers_rerank: int = 3
    num_layers_rerank_fusion: int = 1
    top_k_regions: int = 16
    num_layers_spatial_writer: int = 3
    num_layers_spatial_self: int = 1
    num_layers_spatial_cross: int = 1

    memory_size: int = 12
    val_memory_size: int = 96
    val_vis_delta: float = 0.9
    random_memory_mask_drop: int = 0
    lambda_point: float = 5.0
    lambda_vis: float = 1.0
    lambda_offset: float = 1.0
    lambda_uncertainty: float = 1.0
    lambda_top_k: float = 1.0

    epoch_num: int = 4
    lr: float = 1e-3
    wd: float = 1e-4
    bs: int = 1
    gradient_acc_steps: int = 1
    validation: bool = False
    checkpoint_path: str = "./checkpoints/track_on_checkpoint.pt"
    seed: int = 1234
    loss_after_query: bool = True
    gpus: int = torch.cuda.device_count()
