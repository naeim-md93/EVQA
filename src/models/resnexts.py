import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import BasicBlock, Bottleneck, _log_api_usage_once, conv1x1, load_state_dict_from_url, model_urls


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        model = resnext101_32x8d(pretrained=True)
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

    def forward(self, x):  # (N, 2048, 7, 7)

        x = x / (x.norm(p=2, dim=1, keepdim=True).expand_as(x) + 1e-8)

        return x  # (N, 2048, 7, 7)


class QuestionModel(nn.Module):
    def __init__(self):
        super(QuestionModel, self).__init__()

        # Word Processing
        self.WSA = nn.MultiheadAttention(embed_dim=300, num_heads=5, dropout=0.5, batch_first=True)
        self.drop = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()

        # Sentence Processing
        self.LSTM = nn.LSTM(
            input_size=300,
            hidden_size=1024,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x, mask2d, mask3d):  # (N, T)

        # Word Processing:
        word, _ = self.WSA(x, x, x, key_padding_mask=mask2d)  # (N, T, 300), (N, T, T)
        # print(torch.mean(word, dim=-1))

        word = torch.masked_fill(input=word, mask=mask3d, value=0)  # (N, T, 300)
        word = self.drop(word)  # (N, T, 300)
        word = self.tanh(word)  # (N, T, 300)

        # Question Processing
        _, (_, cn) = self.LSTM(word)  # (2, N, 1024)
        cn = cn.transpose(dim0=1, dim1=0)  # (N, 2, 1024)
        cn = torch.flatten(input=cn, start_dim=1)  # (N, 2048)
        return cn  # (N, 2048)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.image_model = ImageModel()
        self.question_model = QuestionModel()
        self.attention = Attention(
            v_features=2048,
            q_features=2048,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )

        self.classifier = Classifier(
            in_features=2 * 2048 + 2048,
            mid_features=2048,
            out_features=1000,
            drop=0.5,
        )

    def forward(self, img_features, questions):  # (N, 2048, 7, 7), (N, T, 300)

        mask3d = (questions == 0)
        mask2d = ((torch.mean(input=questions, dim=-1)) == 0)
        # print(f'mask2d: {mask2d}')

        # Image Processing
        img_fea = self.image_model(img_features)  # (N, 2048, 7, 7)
        print(img_fea.size())

        # Text Processing
        que_fea = self.question_model(questions, mask2d, mask3d)  # (N, 2048)

        att = self.attention(img_fea, que_fea)  # (N, 2, 7, 7)
        img_fea = apply_attention(img_fea, att)  # (N, 4096)

        combined = torch.cat([img_fea, que_fea], dim=1)  # (N, 4096 + 2048)
        answer = self.classifier(combined)  # (N, 1000)

        return answer  # (N, 1000)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout(p=drop)
        self.lin1 = nn.Linear(in_features=in_features, out_features=mid_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(p=drop)
        self.lin2 = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, x):
        x = self.drop1(x)  # (N, 4096 + 2048)
        x = self.lin1(x)  # (N, 2048)
        x = self.relu(x)  # (N, 2048)
        x = self.drop2(x)  # (N, 2048)
        x = self.lin2(x)  # (N, 1000)
        return x


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(in_channels=v_features, out_channels=mid_features, kernel_size=1, bias=False)
        self.q_lin = nn.Linear(in_features=q_features, out_features=mid_features)
        self.x_conv = nn.Conv2d(in_channels=mid_features, out_channels=glimpses, kernel_size=1)

        self.drop = nn.Dropout(p=drop)
        self.relu = nn.ReLU()

    def forward(self, v, q):
        v = self.drop(v)  # (N, 2048, 7, 7)
        v = self.v_conv(v)  # (N, 512, 7, 7)

        q = self.drop(q)  # (N, 2048)
        q = self.q_lin(q)  # (N, 512)
        q = tile_2d_over_nd(q, v)  # (N, 512, 7, 7)
        x = self.relu(v + q)  # (N, 512, 7, 7)
        x = self.drop(x)  # (N, 512, 7, 7)
        x = self.x_conv(x)  # (N, 2, 7, 7)
        return x  # (N, 2, 7, 7)


def apply_attention(input, attention):  # (N, 2048, 7, 7), (N, 2, 7, 7)
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]  # N, 2048
    glimpses = attention.size(1)  # 2

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1)  # (N, 1, 2048, 49)
    attention = attention.view(n, glimpses, -1)  # (N, 2, 49)
    attention = F.softmax(input=attention, dim=-1).unsqueeze(2)  # (N, 2, 1, 49)
    weighted = attention * input  # (N, 2, 2048, 49)
    weighted_mean = weighted.sum(dim=-1)  # (N, 2, 2048)
    return weighted_mean.view(n, -1)  # (N, 4096)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()  # (N, 512)
    spatial_size = feature_map.dim() - 2  # = 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)  # (N, 512, 7, 7)
    return tiled  # (N, 512, 7, 7)
