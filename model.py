import torch
from torch import nn

class myModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.snps = config.snps
        self.num_classes = config.num_classes
        self.in_channels = config.in_channels
        self.kernel_sizes = config.kernel_sizes

        self.model1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=self.kernel_sizes['conv1'],
                ),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_sizes['pool1']),
            nn.Conv1d(
                in_channels=64,
                out_channels=256,
                kernel_size=self.kernel_sizes['conv1'],
                ),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_sizes['pool1']),
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=self.kernel_sizes['conv1'],
                ),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_sizes['pool1']),
            nn.Flatten()
        )

        input_data = torch.randn(1, self.in_channels, self.snps)
        output_size = self.get_last_pooling_output_size(input_data)

        self.model2 = nn.Sequential(
            nn.Linear(in_features=output_size, out_features=1024),
            nn.Dropout(0.6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=self.num_classes),
            # nn.Sigmoid()
        )


    def get_last_pooling_output_size(self, input_data):
        with torch.no_grad():
            output = self.model1(input_data)
            output_size = output.view(output.size(0), -1).size(1)
        return output_size


    def forward(self, x):
        x = x.view(-1, self.in_channels, self.snps)
        x = self.model1(x)
        x = self.model2(x)
        return x
