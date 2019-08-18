import torch
from torch import nn


# Create LeNet Class
class LeNet(nn.Module):
    def __init__(self):
        # Initialize the super class
        super(LeNet, self).__init__()

        # Define model weights
        # We use Convolution + MaxPooling for each layer
        self.conv_1 = nn.Conv2d(1, 20, 5, padding=2)  # input channels: 3, output channels: 20, kernel size: 5
        # padding = (kernel size - 1) / 2
        self.mp_1 = nn.MaxPool2d(2)  # MaxPooling 2X2

        self.conv_2 = nn.Conv2d(20, 50, 3, padding=1)  # input channels: 20, output channels: 50, kernel size: 3
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(50, 100, 3, padding=1)  # input channels: 50, output channels: 100, kernel size: 3
        self.mp_3 = nn.MaxPool2d(2)

        self.dense_1 = nn.Linear(144 * 100, 500)  # Fully-connected layer, 500 outputs
        # input features = (32/2**(num MaxPool layers))*(num previous output channels)
        self.dense_2 = nn.Linear(500, 7)

    def forward(self, x):
        # layer 1
        h = self.conv_1(x)  # Apply convolution to input
        h = torch.relu(h)  # Apply ReLu nonlinearity
        h = self.mp_1(h)  # Apply MaxPooling
        # layer 2
        h = self.conv_2(h)
        h = torch.relu(h)
        h = self.mp_2(h)
        # layer 3
        h = self.conv_3(h)
        h = torch.relu(h)
        h = self.mp_3(h)
        # flatten
        h = h.view(x.shape[0], -1)  # Reshape images to (batch size)X(num features)
        # hidden
        h = self.dense_1(h)  # Apply linear hidden layer
        h = torch.relu(h)  # Apply ReLu nonlinearity
        # output
        logits = self.dense_2(h)  # Apply linear layer
        # NO ACTIVATION NEEDED HERE!
        return logits


# Create LeNet Class
class BN_LeNet(nn.Module):
    def __init__(self):
        # Initialize the super class
        super(BN_LeNet, self).__init__()

        # Define model weights
        # We use Convolution + MaxPooling for each layer
        self.conv_1 = nn.Conv2d(1, 20, 5, padding=2)  # input channels: 3, output channels: 20, kernel size: 5
        # padding = (kernel size - 1) / 2
        self.bn_1 = nn.BatchNorm2d(20)  # Batch Normalization
        self.mp_1 = nn.MaxPool2d(2)  # MaxPooling 2X2

        self.conv_2 = nn.Conv2d(20, 50, 3, padding=1)  # input channels: 20, output channels: 50, kernel size: 3
        self.bn_2 = nn.BatchNorm2d(50)  # Batch Normalization
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(50, 100, 3, padding=1)  # input channels: 50, output channels: 100, kernel size: 3
        self.bn_3 = nn.BatchNorm2d(100)  # Batch Normalization
        self.mp_3 = nn.MaxPool2d(2)

        self.dense_1 = nn.Linear(144 * 100, 500)  # Fully-connected layer, 500 outputs
        # input features = (32/2**(num MaxPool layers))*(num previous output channels)
        self.bn_4 = nn.BatchNorm1d(500)  # Batch Normalization
        self.dense_2 = nn.Linear(500, 7)

    def forward(self, x):
        # layer 1
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = torch.relu(h)
        h = self.mp_1(h)
        # layer 2
        h = self.conv_2(h)
        h = self.bn_2(h)
        h = torch.relu(h)
        h = self.mp_2(h)
        # layer 3
        h = self.conv_3(h)
        h = self.bn_3(h)
        h = torch.relu(h)
        h = self.mp_3(h)
        # flatten
        h = h.view(x.shape[0], -1)
        # hidden
        h = self.dense_1(h)
        h = self.bn_4(h)
        h = torch.relu(h)
        logits = self.dense_2(h)
        return logits


class MLPConv(nn.Module):

    def __init__(self, inputs, num_conv, num_h1, num_h2):
        super(MLPConv, self).__init__()

        self.conv_1 = nn.Conv2d(inputs, num_conv, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_conv)
        self.conv_2 = nn.Conv2d(num_conv, num_h1, 1)
        self.bn_2 = nn.BatchNorm2d(num_h1)
        self.conv_3 = nn.Conv2d(num_h1, num_h2, 1)
        self.bn_3 = nn.BatchNorm2d(num_h2)

    def forward(self, x):
        h = torch.relu(self.bn_1(self.conv_1(x)))
        h = torch.relu(self.bn_2(self.conv_2(h)))
        return self.bn_3(self.conv_3(h))


class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()

        self.conv_1 = MLPConv(1, 10, 20, 20)
        self.mp_1 = nn.MaxPool2d(2)

        self.conv_2 = MLPConv(20, 30, 50, 50)
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = MLPConv(50, 70, 100, 100)
        self.mp_3 = nn.MaxPool2d(2)

        self.dense_1 = nn.Linear(144 * 100, 500)
        self.bn = nn.BatchNorm1d(500)
        self.dense_2 = nn.Linear(500, 10)

    def forward(self, x):
        # layer 1
        h = self.conv_1(x)
        h = torch.relu(h)
        h = self.mp_1(h)
        # layer 2
        h = self.conv_2(h)
        h = torch.relu(h)
        h = self.mp_2(h)
        # layer 3
        h = self.conv_3(h)
        h = torch.relu(h)
        h = self.mp_3(h)
        # flatten
        h = h.view(x.shape[0], -1)
        # hidden
        h = self.dense_1(h)
        h = self.bn(h)
        h = torch.relu(h)
        # output
        logits = self.dense_2(h)
        return logits


class Inception(nn.Module):

    def __init__(self, input_chanel, output_chanel):
        super(Inception, self).__init__()

        self.c1 = nn.Conv2d(input_chanel, output_chanel, 1)
        self.bn_1 = nn.BatchNorm2d(output_chanel)

        self.c21 = nn.Conv2d(input_chanel, output_chanel, 1)
        self.bn_21 = nn.BatchNorm2d(output_chanel)
        self.c22 = nn.Conv2d(output_chanel, output_chanel, 3, padding=1)
        self.bn_22 = nn.BatchNorm2d(output_chanel)

        self.c31 = nn.Conv2d(input_chanel, output_chanel, 1)
        self.bn_31 = nn.BatchNorm2d(output_chanel)
        self.c32 = nn.Conv2d(output_chanel, output_chanel, 5, padding=2)
        self.bn_32 = nn.BatchNorm2d(output_chanel)

        self.m41 = nn.MaxPool2d(3, 1, padding=1)
        self.c42 = nn.Conv2d(input_chanel, output_chanel, 1)
        self.bn_42 = nn.BatchNorm2d(output_chanel)

    def forward(self, x):
        h1 = torch.relu(self.bn_1(self.c1(x)))

        h2 = torch.relu(self.bn_21(self.c21(x)))
        h2 = torch.relu(self.bn_22(self.c22(h2)))

        h3 = torch.relu(self.bn_31(self.c31(x)))
        h3 = torch.relu(self.bn_32(self.c32(h3)))

        h4 = self.m41(x)
        h4 = torch.relu(self.bn_42(self.c42(h4)))

        h = torch.cat((h1, h2, h3, h4), 1)

        return h


class InceptionV1(nn.Module):

    def __init__(self):
        super(InceptionV1, self).__init__()

        self.c1 = Inception(1, 4)
        self.c2 = Inception(16, 8)
        self.c3 = Inception(32, 16)

        self.m1 = nn.MaxPool2d(2)
        self.m2 = nn.MaxPool2d(2)
        self.m3 = nn.MaxPool2d(2)

        self.l1 = nn.Linear(9216, 500)
        self.bn = nn.BatchNorm1d(500)
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = self.m1(h)
        h = torch.relu(self.c2(h))
        h = self.m2(h)
        h = torch.relu(self.c3(h))
        h = self.m3(h)
        h = h.view(h.shape[0], -1)
        h = torch.relu(self.bn(self.l1(h)))
        h = self.l2(h)
        return h


class Residual(nn.Module):
    def __init__(self, inner):
        super(Residual, self).__init__()

        self.inner = inner

    def forward(self, x):
        h = self.inner(x)
        return h + x


class ResNetBlock(nn.Module):
    def __init__(self, num_inputs):
        super(ResNetBlock, self).__init__()

        self.conv_1 = nn.Conv2d(num_inputs, num_inputs, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_inputs)

        self.conv_2 = nn.Conv2d(num_inputs, num_inputs, 3, padding=1)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv_1(x)))
        return self.conv_2(h)


class ResNet(nn.Module):
    def __init__(self, num_residual):
        super(ResNet, self).__init__()

        self.conv_1 = nn.Conv2d(1, 20, 7, padding=3)
        self.bn_1 = nn.BatchNorm2d(20)
        # define list of residual blocks
        self.res_block_1 = [(Residual(ResNetBlock(20)), nn.BatchNorm2d(20)) for i in range(num_residual)]
        # PyTorch does not support lists of modules,
        # so every module should be manually added to network parameters.
        for i, (conv, bn) in enumerate(self.res_block_1):
            self.add_module("res_conv_1_" + str(i), conv)
            self.add_module("res_bn_1_" + str(i), bn)
        self.mp_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(20, 35, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(35)
        self.res_block_2 = [(Residual(ResNetBlock(35)), nn.BatchNorm2d(35)) for i in range(num_residual)]
        for i, (conv, bn) in enumerate(self.res_block_2):
            self.add_module("res_conv_2_" + str(i), conv)
            self.add_module("res_bn_2_" + str(i), bn)
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(35, 50, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(50)
        self.res_block_3 = [(Residual(ResNetBlock(50)), nn.BatchNorm2d(50)) for i in range(num_residual)]
        for i, (conv, bn) in enumerate(self.res_block_3):
            self.add_module("res_conv_3_" + str(i), conv)
            self.add_module("res_bn_3_" + str(i), bn)
        self.mp_3 = nn.MaxPool2d(2)

        self.l1 = nn.Linear(144 * 50, 500)
        self.bn = nn.BatchNorm1d(500)
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        h = self.bn_1(self.conv_1(x))
        h = torch.relu(h)
        for (conv, bn) in self.res_block_1:
            h = torch.relu(bn(conv(h)))
        h = self.mp_1(h)

        h = self.bn_2(self.conv_2(h))
        h = torch.relu(h)
        for (conv, bn) in self.res_block_2:
            h = torch.relu(bn(conv(h)))
        h = self.mp_2(h)

        h = self.bn_3(self.conv_3(h))
        h = torch.relu(h)
        for (conv, bn) in self.res_block_3:
            h = torch.relu(bn(conv(h)))
        h = self.mp_3(h)
        # flatten
        h = h.view(x.shape[0], -1)
        # hidden
        h = self.l1(h)
        h = self.bn(h)
        h = torch.relu(h)
        # output
        logits = self.l2(h)
        return logits


class DenseBlock(nn.Module):
    def __init__(self, num_conv):
        super(DenseBlock, self).__init__()

        self.conv_1 = nn.Conv2d(num_conv, num_conv, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_conv)
        self.conv_2 = nn.Conv2d(num_conv, num_conv, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_conv)
        self.conv_3 = nn.Conv2d(num_conv, num_conv, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_conv)
        self.conv_4 = nn.Conv2d(num_conv, num_conv, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(num_conv)

    def forward(self, x):
        h1 = torch.relu(self.bn_1(self.conv_1(x)))
        h2 = torch.relu(self.bn_2(self.conv_2(h1 + x)))
        h3 = torch.relu(self.bn_3(self.conv_3(x + h1 + h2)))
        h4 = torch.relu(self.bn_4(self.conv_4(x + h1 + h2 + h3)))
        return h4 + h3 + h2 + h1 + x


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.conv_1 = nn.Conv2d(1, 20, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(20)
        self.db_1 = DenseBlock(20)
        self.mp_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(20, 50, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(50)
        self.db_2 = DenseBlock(50)
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(50, 100, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(100)
        self.db_3 = DenseBlock(100)
        self.mp_3 = nn.MaxPool2d(2)

        self.dense_1 = nn.Linear(144 * 100, 500)
        self.bn = nn.BatchNorm1d(500)
        self.dense_2 = nn.Linear(500, 10)

    def forward(self, x):
        # layer 1
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = torch.relu(h)
        h = self.db_1(h)
        h = self.mp_1(h)

        # layer 2
        h = self.bn_2(self.conv_2(h))
        h = torch.relu(h)
        h = self.db_2(h)
        h = self.mp_2(h)

        # layer 3
        h = self.bn_3(self.conv_3(h))
        h = torch.relu(h)
        h = self.db_3(h)
        h = self.mp_3(h)

        # flatten
        h = h.view(x.shape[0], -1)
        # hidden
        h = self.bn(self.dense_1(h))
        h = torch.relu(h)
        # output
        logits = self.dense_2(h)
        return logits


class XConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, padding=1):
        super(XConv, self).__init__()

        self.pointwise = nn.Conv2d(input_size, output_size, 1)

        # Specify group convolution with groups=num_channel
        self.depthwise = nn.Conv2d(output_size, output_size, kernel_size, padding=padding, groups=output_size)
        self.bn_2 = nn.BatchNorm2d(output_size)

    def forward(self, x):
        h = self.pointwise(x)
        h = self.bn_2(self.depthwise(h))
        return h


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        self.c1 = XConv(1, 10)
        self.c2 = XConv(10, 20)

        self.sc1 = nn.Conv2d(1, 20, 1, 2)

        self.c3 = XConv(20, 35)
        self.c4 = XConv(35, 50)

        self.sc2 = nn.Conv2d(20, 50, 1, 2)

        self.c5 = XConv(50, 75)
        self.c6 = XConv(75, 100)

        self.sc3 = nn.Conv2d(50, 100, 1, 2)

        self.c7 = XConv(100, 100)
        self.c8 = XConv(100, 100)

        self.m1 = nn.MaxPool2d(2)
        self.m2 = nn.MaxPool2d(2)
        self.m3 = nn.MaxPool2d(2)

        self.l1 = nn.Linear(144 * 100, 500)
        self.bn = nn.BatchNorm1d(500)
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(h))
        h = self.m1(h)
        x = self.sc1(x)
        h = h + x
        h = torch.relu(self.c3(h))
        h = torch.relu(self.c4(h))
        h = self.m2(h)
        x = self.sc2(x)
        h = h + x
        h = torch.relu(self.c5(h))
        h = torch.relu(self.c6(h))
        h = self.m3(h)
        x = self.sc3(x)
        h = h + x
        h = torch.relu(self.c7(h))
        h = torch.relu(self.c8(h))
        h = h.view(h.shape[0], -1)
        h = torch.relu(self.bn(self.l1(h)))
        h = self.l2(h)
        return h


class LinearBottleneck(nn.Module):
    def __init__(self, input_shape, expansion_factor, output_shape=None, stride=1):
        super(LinearBottleneck, self).__init__()

        self.activation = nn.ReLU6()

        num_layers = input_shape * expansion_factor
        output_shape = output_shape or input_shape

        self.conv_1 = nn.Conv2d(input_shape, num_layers, 1)
        self.conv_2 = nn.Conv2d(num_layers, num_layers, 3, stride, 1, groups=num_layers)
        self.conv_3 = nn.Conv2d(num_layers, output_shape, 1)

        self.bn_1 = nn.BatchNorm2d(num_layers)
        self.bn_2 = nn.BatchNorm2d(num_layers)
        self.bn_3 = nn.BatchNorm2d(output_shape)

    def forward(self, x):
        # pointwise
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = self.activation(h)
        # depthwise
        h = self.conv_2(h)
        h = self.bn_2(h)
        h = self.activation(h)
        # pointwise
        h = self.conv_3(h)
        h = self.bn_3(h)
        return h


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.activation = nn.ReLU6()

        self.conv_0 = LinearBottleneck(1, 5, 20)

        self.conv_1 = Residual(LinearBottleneck(20, 5))
        self.conv_2 = Residual(LinearBottleneck(20, 5))

        self.conv_3 = LinearBottleneck(20, 5, 50, 2)

        self.conv_4 = Residual(LinearBottleneck(50, 5))
        self.conv_5 = Residual(LinearBottleneck(50, 5))

        self.conv_6 = LinearBottleneck(50, 5, 100, 2)

        self.conv_7 = Residual(LinearBottleneck(100, 5))
        self.conv_8 = Residual(LinearBottleneck(100, 5))

        self.conv_9 = LinearBottleneck(100, 5, 200, 2)

        self.pool = nn.AvgPool2d(8)

        self.dense_1 = nn.Linear(200, 500)
        self.bn = nn.BatchNorm1d(500)
        self.dense_2 = nn.Linear(500, 10)

    def forward(self, x):
        h = self.activation(self.conv_0(x))
        h = self.activation(self.conv_1(h))
        h = self.activation(self.conv_2(h))
        h = self.activation(self.conv_3(h))
        h = self.activation(self.conv_4(h))
        h = self.activation(self.conv_5(h))
        h = self.activation(self.conv_6(h))
        h = self.activation(self.conv_7(h))
        h = self.activation(self.conv_8(h))
        h = self.activation(self.conv_9(h))
        h = self.pool(h).view(x.size()[0], -1)
        h = self.activation(self.dense_1(h))
        h = self.bn(h)
        return self.dense_2(h)


class Bottleneck(nn.Module):
    def __init__(self, input_shape, expansion_factor, cardinality=None, output_shape=None, stride=1):
        super(Bottleneck, self).__init__()

        self.activation = nn.ReLU6()

        num_layers = input_shape // expansion_factor
        output_shape = output_shape or input_shape
        cardinality = cardinality or 1

        self.conv_1 = nn.Conv2d(input_shape, num_layers, 1)
        self.conv_2 = nn.Conv2d(num_layers, num_layers, 3, stride, 1, groups=cardinality)
        self.conv_3 = nn.Conv2d(num_layers, output_shape, 1)

        self.bn_1 = nn.BatchNorm2d(num_layers)
        self.bn_2 = nn.BatchNorm2d(num_layers)
        self.bn_3 = nn.BatchNorm2d(output_shape)

    def forward(self, x):
        # pointwise
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = self.activation(h)
        # depthwise
        h = self.conv_2(h)
        h = self.bn_2(h)
        h = self.activation(h)
        # pointwise
        h = self.conv_3(h)
        h = self.bn_3(h)
        return h


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()

        self.activation = nn.ReLU6()

        self.conv_0 = Bottleneck(1, 1, output_shape=20)

        self.conv_1 = Residual(Bottleneck(20, 1, 5))
        self.conv_2 = Residual(Bottleneck(20, 1, 5))

        self.conv_3 = Bottleneck(20, 1, output_shape=50, stride=2)

        self.conv_4 = Residual(Bottleneck(50, 1, 10))
        self.conv_5 = Residual(Bottleneck(50, 1, 10))

        self.conv_6 = Bottleneck(50, 1, output_shape=100, stride=2)

        self.conv_7 = Residual(Bottleneck(100, 1, 20))
        self.conv_8 = Residual(Bottleneck(100, 1, 20))

        self.conv_9 = Bottleneck(100, 1, output_shape=200, stride=2)

        self.pool = nn.AvgPool2d(8)

        self.dense_1 = nn.Linear(200, 500)
        self.bn = nn.BatchNorm1d(500)
        self.dense_2 = nn.Linear(500, 10)

    def forward(self, x):
        h = self.activation(self.conv_0(x))
        h = self.activation(self.conv_1(h))
        h = self.activation(self.conv_2(h))
        h = self.activation(self.conv_3(h))
        h = self.activation(self.conv_4(h))
        h = self.activation(self.conv_5(h))
        h = self.activation(self.conv_6(h))
        h = self.activation(self.conv_7(h))
        h = self.activation(self.conv_8(h))
        h = self.activation(self.conv_9(h))
        h = self.pool(h).view(x.size()[0], -1)
        h = self.activation(self.dense_1(h))
        h = self.bn(h)
        return self.dense_2(h)


class SEBlock(nn.Module):
    def __init__(self, num_channels):
        super(SEBlock, self).__init__()

        self.lin1 = nn.Conv2d(num_channels, num_channels, 1)
        self.lin2 = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x):
        h = nn.functional.avg_pool2d(x, int(x.size()[2]))

        h = torch.relu(self.lin1(h))
        h = torch.sigmoid(self.lin2(h))

        return x * h


class SE_LeNet(nn.Module):
    def __init__(self):
        super(SE_LeNet, self).__init__()

        self.conv_1 = nn.Conv2d(1, 20, 5, padding=2)
        self.bn_1 = nn.BatchNorm2d(20)
        self.se_1 = SEBlock(20)
        self.mp_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(20, 50, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(50)
        self.se_2 = SEBlock(50)
        self.mp_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Conv2d(50, 100, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(100)
        self.se_3 = SEBlock(100)
        self.mp_3 = nn.MaxPool2d(2)

        self.dense_1 = nn.Linear(144 * 100, 500)
        self.bn_4 = nn.BatchNorm1d(500)
        self.dense_2 = nn.Linear(500, 10)

    def forward(self, x):
        # layer 1
        h = self.conv_1(x)
        h = self.bn_1(h)
        torch.relu_(h)
        h = self.se_1(h)
        h = self.mp_1(h)
        # layer 2
        h = self.conv_2(h)
        h = self.bn_2(h)
        torch.relu_(h)
        h = self.se_2(h)
        h = self.mp_2(h)
        # layer 3
        h = self.conv_3(h)
        h = self.bn_3(h)
        torch.relu_(h)
        h = self.se_3(h)
        h = self.mp_3(h)
        # flatten
        h = h.view(x.shape[0], -1)
        # hidden
        h = self.dense_1(h)
        h = self.bn_4(h)
        torch.relu_(h)
        logits = self.dense_2(h)
        return logits

