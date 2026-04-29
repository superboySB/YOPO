import time
import torch
import torch.nn
from policy.models.resnet import resnet18


# input: [C, 96, 160]
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int, input_channels: int = 1):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [1, 32, 64])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int, input_channels: int = 1):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim, input_channels=1):
    return ResNet18(output_dim, input_channels=input_channels)


if __name__ == '__main__':
    net = YopoBackbone(64, 3)
    input_ = torch.zeros((1, 3, 96, 160))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
