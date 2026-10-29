from dataclasses import dataclass

from torch import nn

from ..blocks.conv import ConvTransposeBlock, ConvTransposeBlockConfig


@dataclass(frozen=True)
class GeneratorConfig:
    latent_dim: int
    convT_configs: tuple[ConvTransposeBlockConfig, ...]


class Generator(nn.Sequential):
    def __init__(self, config: GeneratorConfig) -> None:
        blocks: list[nn.Module] = [nn.Unflatten(1, (config.latent_dim, 1, 1))]
        in_channels = config.latent_dim
        for block_conf in config.convT_configs:
            blocks.append(ConvTransposeBlock(in_channels, block_conf))
            in_channels = block_conf.out_channels

        super().__init__(*blocks)
        self.config = config
