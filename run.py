import torch
from tensorboardX import SummaryWriter
from torchinfo import summary
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from utils import set_random_seed
from config import myConfig
from train import train_one_epoch, val_one_epoch, test_one_epoch


def main():
    seed_num = 42
    set_random_seed(seed_num)

    config = myConfig()

    writer = SummaryWriter(comment=myConfig.comment)

    summary(config.model, input_size=(1, config.ModelConfig.in_channels, config.ModelConfig.snps))
    writer.add_graph(config.model, input_to_model=torch.randn(1, config.ModelConfig.in_channels, config.ModelConfig.snps).to(config.device))

    epoch_num = 1000
    val_interval = 10
    test_interval = 10

    # 打印是否运行在GPU上
    print(f'device: {config.device}')

    for epoch in range(1, epoch_num + 1):
        train_one_epoch(epoch, writer, config)
        if epoch % val_interval == 0:
            val_one_epoch(epoch, writer, config)
        if epoch % test_interval == 0:
            test_one_epoch(epoch, writer, config)

    writer.close()

if __name__ == "__main__":
    main()
