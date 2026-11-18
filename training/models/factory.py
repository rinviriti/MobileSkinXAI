"""Exact inference backbones; enhancement/control add no inference modules."""
import timm
ARCHITECTURES={
 'MobileNetV3-Small':'mobilenetv3_small_100.lamb_in1k',
 'EfficientNet-Lite0':'tf_efficientnet_lite0.in1k',
 'MobileNetV4-Conv-Small':'mobilenetv4_conv_small.e2400_r224_in1k',
 'ConvNeXt-Tiny':'convnext_tiny.fb_in1k',
 'MobileSkinXAI-Enhanced':'mobilenetv4_conv_small.e2400_r224_in1k',
 'MobileSkinXAI-Enhanced (LR Control)':'mobilenetv4_conv_small.e2400_r224_in1k',
}
CLASSES=('NV','MEL','BCC','SCC')
def create_model(name,pretrained=False):
 return timm.create_model(ARCHITECTURES[name],pretrained=pretrained,num_classes=4)
