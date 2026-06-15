from dataclasses import dataclass, field


@dataclass
class UNetConfig:
    sample_size: int = 128
    in_channels: int = 6
    out_channels: int = 3
    layers_per_block: int = 3
    block_out_channels: tuple[int, ...] = (32, 64)
    down_block_types: tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D")
    up_block_types: tuple[str, ...] = (
        "AttnUpBlock2D",
        "UpBlock2D",
    )


@dataclass
class TrainingConfig:
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-5
    crop_size: int = 128
    num_workers: int = 2
    save_dir: str = "experiments_v1"
    viz_interval: int = 25
    num_steps_generation: int = 20
