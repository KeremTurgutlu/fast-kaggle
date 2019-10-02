#AUTOGENERATED! DO NOT EDIT! File to edit: dev/13_segmentation_models.ipynb (unless otherwise specified).

__all__ = ['default_configs', 'model_funcs', 'splits', 'get_model', 'dunet_config', 'resdunet18', 'resdunet34',
           'resdunet50', 'resdunet101', 'resdunet152', 'seresdunext50', 'seresdunext101', 'deeplab_config',
           'deeplabv3res50', 'deeplabv3res101', 'deeplabv3seresnext50', 'deeplabv3seresnext101']

#Cell
from fastai.vision import *

#Cell
default_configs = {}
model_funcs = {}
splits = {}

#Cell
def get_model(name, data, config):
    "Get model given name, data and config. Undefined config is defaulted."
    conf, copy_conf = default_configs[name].copy(), default_configs[name].copy()
    conf.update(config)
    f = model_funcs[name]
    model = f(data, conf)
    split_fn = splits.get(name)
    return model, split_fn, copy_conf

#Cell
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import cnn_config

_body_config = {"pretrained":True}
_unet_config = {"blur":False, "blur_final":True, "self_attention":False,
         "y_range":None, "norm_type":NormType, "last_cross":True, "bottle":False}
dunet_config = {**_body_config, **_unet_config}

#Cell
from fastai.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from fastai.vision.models.cadene_models import model_meta

#Cell
_res_meta = model_meta[resnet18]
_res_cut, _res_split = _res_meta['cut'], _res_meta['split']

#Cell
def _resdunet(arch, data, config):
    "Returns a resdunet model for a arch from data and final config"
    pretrained = config.pop("pretrained")
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    body = create_body(arch, pretrained, _res_cut)
    model = DynamicUnet(body, n_classes=data.c, img_size=size, **config)
    return model

#Cell
def resdunet18(data, config): return _resdunet(resnet18, data, config)
model_funcs['resdunet18'] = resdunet18
default_configs['resdunet18'] = dunet_config
splits['resdunet18'] = _res_split

def resdunet34(data, config): return _resdunet(resnet34, data, config)
model_funcs['resdunet34'] = resdunet34
default_configs['resdunet34'] = dunet_config
splits['resdunet34'] = _res_split

def resdunet50(data, config): return _resdunet(resnet50, data, config)
model_funcs['resdunet50'] = resdunet50
default_configs['resdunet50'] = dunet_config
splits['resdunet50'] = _res_split

def resdunet101(data, config): return _resdunet(resnet101, data, config)
model_funcs['resdunet101'] = resdunet101
default_configs['resdunet101'] = dunet_config
splits['resdunet101'] = _res_split

def resdunet152(data, config): return _resdunet(resnet152, data, config)
model_funcs['resdunet152'] = resdunet152
default_configs['resdunet152'] = dunet_config
splits['resdunet152'] = _res_split

#Cell
from fastai.vision.models.cadene_models import se_resnext50_32x4d, se_resnext101_32x4d

#Cell
_se_meta = model_meta[se_resnext50_32x4d]
_se_cut, _se_split = _se_meta['cut'], _se_meta['split']

#Cell
def _seresdunext(arch, data, config):
    "Returns a resdunet model for a arch from data and final config"
    pretrained = config.pop("pretrained")
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    body = create_body(arch, pretrained, cut=_se_cut)
    model = DynamicUnet(body, n_classes=data.c, img_size=size, **config)
    return model

#Cell
def seresdunext50(data, config): return _seresdunext(se_resnext50_32x4d, data, config)
model_funcs['seresdunext50'] = seresdunext50
default_configs['seresdunext50'] = dunet_config
splits['seresdunext50'] = _se_split

def seresdunext101(data, config): return _seresdunext(se_resnext101_32x4d, data, config)
model_funcs['seresdunext101'] = seresdunext101
default_configs['seresdunext101'] = dunet_config
splits['seresdunext101'] = _se_split

#Cell
_body_config = {"pretrained":True}
_deeplab_config = {'variant':'D', 'skip':'m1', 'skip_num':48}
deeplab_config = {**_body_config, **_deeplab_config}

#Cell
class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

def _Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

#Cell
class _DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, backbone='seresnext-50', pretrained=True, variant='D',
                 skip='m1', skip_num=48):
        super(_DeepV3Plus, self).__init__()

        self.variant, self.skip, self.skip_num = variant, skip, skip_num

        if backbone == 'seresnext50':
            body = create_body(se_resnext50_32x4d, pretrained)
        elif backbone == 'seresnext101':
            body = create_body(se_resnext101_32x4d, pretrained)
        elif backbone == 'resnet50':
            body = create_body(resnet50, pretrained)
            body = nn.Sequential(nn.Sequential(body[:4]), *body[4:])
        elif backbone == 'resnet101':
            body = create_body(resnet101, pretrained)
            body = nn.Sequential(nn.Sequential(body[:4]), *body[4:])
        else:
            raise ValueError("Not a valid network arch")

        self.body = body

        if self.variant == 'D':
            for n, m in self.body[3].named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.body[4].named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.body[4].named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        # init weights and biases
        for m in [self.aspp, self.bot_aspp, self.bot_fine, self.final]:
            apply_init(m, nn.init.kaiming_normal_)


    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.body[0](x)  # 400
        x1 = self.body[1](x0)  # 400
        x2 = self.body[2](x1)  # 100
        x3 = self.body[3](x2)  # 100
        x4 = self.body[4](x3)  # 100
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        main_out = _Upsample(dec1, x_size[2:])

        return main_out

#Cell
def _deeplabv3(arch_name, data, config):
    "Returns a resdunet model for a arch from data and final config"
    pretrained = config.pop("pretrained")
    model = _DeepV3Plus(data.c, arch_name, pretrained=pretrained, **config)
    return model

#Cell
def _deeplab_split(m:nn.Module): return (m.body[3], m.aspp)

#Cell
def deeplabv3res50(data, config): return _deeplabv3("resnet50", data, config)
model_funcs['deeplabv3res50'] = deeplabv3res50
default_configs['deeplabv3res50'] = deeplab_config
splits['deeplabv3res50'] = _deeplab_split

def deeplabv3res101(data, config): return _deeplabv3("resnet101", data, config)
model_funcs['deeplabv3res101'] = deeplabv3res101
default_configs['deeplabv3res101'] = deeplab_config
splits['deeplabv3res101'] = _deeplab_split

def deeplabv3seresnext50(data, config): return _deeplabv3("seresnext50", data, config)
model_funcs['deeplabv3seresnext50'] = deeplabv3seresnext50
default_configs['deeplabv3seresnext50'] = deeplab_config
splits['deeplabv3seresnext50'] = _deeplab_split

def deeplabv3seresnext101(data, config): return _deeplabv3("seresnext101", data, config)
model_funcs['deeplabv3seresnext101'] = deeplabv3seresnext101
default_configs['deeplabv3seresnext101'] = deeplab_config
splits['deeplabv3seresnext101'] = _deeplab_split