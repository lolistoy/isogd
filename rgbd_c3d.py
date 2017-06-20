from torch.autograd.variable import Variable
import torch.autograd
import torch.nn as nn
import math
import torch

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, feature_name, dropout):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], force_downsample=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.dropout = dropout
        self.dropout1 = nn.Dropout3d(dropout)
        self.dropout2 = nn.Dropout3d(dropout)
        self.dropout3 = nn.Dropout3d(dropout)
        self.dropout4 = nn.Dropout3d(dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.feature_name = feature_name
    def _make_layer(self, block, planes, blocks, stride=1, force_downsample=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or force_downsample:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.feature_name == 'layer2':
            return x
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        #x = self.avgpool(x)

        return x


def resnet18_c3d(load_pretrained=False, feature_name='layer4',  dropout=0.5, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert feature_name == 'layer2' or feature_name == 'layer4'
    model = ResNet(BasicBlock, [2, 2, 2, 2], feature_name=feature_name,  dropout=dropout, **kwargs)
    if load_pretrained:
        model.load_state_dict(torch.load('./model/resnet18_c3d.model'))
    return model


class RGBDC3D(nn.Module):
    def __init__(self, cnn_name='resnet18_c3d', feature_name='layer4', cnn_dropout=0.5,
                 modality='rgb',
                 seq_len=8, n_class=249, gpu_id=None):
        super(RGBDC3D, self).__init__()
        self.cnn_name = cnn_name
        assert cnn_name in ['resnet18_c3d']
        self.modality = modality
        self.seq_len = seq_len
        self.n_class = n_class
        self.feature_name = feature_name
        self.gpu_id = gpu_id
        config = locals()
        config['self'] = self.__class__.__name__
        self.config = config

        if cnn_name == 'resnet18_c3d':
            self.cnn = resnet18_c3d(load_pretrained=True, feature_name=feature_name, dropout=cnn_dropout)
            fc_dim = 512
        else:
            raise RuntimeError('cnn_name wrong!')

        kernel_size1 = (4, 1, 1)
        kernel_size2 = (1, 7, 7)
        self.classifier_conv3d1 = nn.Sequential(nn.Dropout3d(0.), nn.AvgPool3d((1, 7, 7)),
                                                nn.Conv3d(512, self.n_class, kernel_size=kernel_size1))

        self.classifier_conv3d2 = nn.Sequential(nn.Dropout3d(0.), nn.AvgPool3d((4, 1, 1)),
                                                nn.Conv3d(512, n_class, kernel_size=kernel_size2))

    def forward(self, inps):
        imgs = inps[0]
        imgs = imgs.transpose(1, 2)
        imgs = Variable(imgs, volatile=not self.training)
        feats = nn.parallel.data_parallel(self.cnn, imgs, self.gpu_id)
        ys1 = nn.parallel.data_parallel(self.classifier_conv3d1, feats, self.gpu_id)
        ys2 = nn.parallel.data_parallel(self.classifier_conv3d2, feats, self.gpu_id)
        ys = ys1 + ys2
        if torch.__version__ == '0.1.12_2':
            ys = ys.mean(2).mean(3).mean(4).view(-1, self.n_class)
        else:
            ys = ys.mean(2, True).mean(3, True).mean(4, True).view(-1, self.n_class)
        output = (ys, )
        return output
