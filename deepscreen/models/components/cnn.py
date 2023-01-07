from torch import nn, rand
from torch.autograd import Variable


class CNN(nn.Sequential):
    def __init__(
            self,
            filters: list[int],
            kernels: list[int],
            max_sequence_length: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()
        num_layer = len(filters)
        channels = [in_channels] + filters
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=channels[i],
                                             out_channels=channels[i+1],
                                             kernel_size=kernels[i])
                                   for i in range(num_layer)])
        n_size = self._get_conv_output((in_channels, max_sequence_length))
        self.fc1 = nn.Linear(n_size, out_channels)

    def _forward_features(self, x):
        for layer in self.conv:
            x = nn.functional.relu(layer(x))
        x = nn.functional.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v)
        v = v.view(v.size(0), -1)
        v = self.fc1(v)
        return v

    def _get_conv_output(self, shape):
        bs = 1
        input_feat = Variable(rand(bs, *shape))
        output_feat = self._forward_features(input_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


