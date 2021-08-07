# detectron2_timm
A Simple wrapper for binding the models in [timm](https://github.com/rwightman/pytorch-image-models) library into 
[detectron2](https://github.com/facebookresearch/detectron2) backbone for training two-stage detectors using `detection2`. This simple library, does not require and changes to either `timm` models or `detectron2`. Whichever model is found in `timm.models.list_models()` is automatically binded to the `detectron2` backbone (including `FPN`).

#### Configs
The configuration of the binded backbone model from `timm` is specified in the extend [config](https://github.com/iKrishneel/detectron2_timm/blob/master/detectron2_timm/config/config.py). In the config you specify laters from which to extract the features, rename this feature extractor layers and also which layers to remove from the `timm` model. 

The extended config has the following attributes:
```yaml
cfg.MODEL.BACKBONE.NAME = "build_xcit_small_12_p8_224_fpn_backbone"       # name of the model
cfg.MODEL.BACKBONE.FREEZE_AT = 0                                          # freeze at which layer

# backbone config
cfg.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS = []                              # layers to remove from the timm model
cfg.MODEL.BACKBONE.CONFIG.OUT_FEATURES = []                               # layers in timm model from which to extract the features
cfg.MODEL.BACKBONE.CONFIG.REMAPS = []                                     # name of the output features, the order must be same as out_features
cfg.MODEL.BACKBONE.CONFIG.STRIDES = []                                    # strides of each output features, the order must be same out_features
cfg.MODEL.BACKBONE.CONFIG.PRETRAINED = False                              # init with pretrained model   

# input
cfg.INPUT.FIXED_INPUT_SIZE = True                                         # model has fix size input, eg. for transformers
```

#### Model Name Scheme
The binding backbone model uses similar naming scheme as in detectron2. Basically `build_` and `_backbone` are appendend before and after the timm model name. Example for the model `xcit_small_12_p8_224` in timm, the binding backbone without FPN name is
```bash
build_xcit_small_12_p8_224_backbone
```
and the model with FPN has `_fpn` appended before `_backbone`
```bash
build_xcit_small_12_p8_224_fpn_backbone
```

#### Using the Backbone
The binded backbone are added the global `BACKBONE_REGISTRY` dict of detectron2. To use any of the model, just import the backbone

```python
# example
from detectron2_timm import build_xcit_small_12_p8_224_fpn_backbone
```

## Dependencies
The library is tested with the following dependencies
```bash
einops == 0.3.0
torch == 1.8.0
torchvision == 0.9.0
timm == 0.4.13   # for xcit models
```

## Train and Evaluation
The training and evaluation are same as that in `detectron2` except that you should use the [train_net.py](https://github.com/iKrishneel/detectron2_timm/blob/master/tools/train_net.py) which will have all the binding models.

#### Train
To train the `xcit` model
```bash
$ python tools/train_net.py --config-file config/xcit/xcit_small_12_p8_224_fpn.yaml --num-gpus 4
```

#### Test
To test the `xcit` model
```bash
$ python tools/train_net.py --config-file config/xcit/xcit_small_12_p8_224_fpn.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS ./logs/mrcnn_xcit_small_12_p8.pth
```

## Examples
Checkout the [xcit.ipynb](https://github.com/iKrishneel/detectron2_timm/blob/master/scripts/xcit.ipynb) which shows example of using this library with the [pretrained xcit models](https://github.com/facebookresearch/xcit/tree/master/detection) for evaluation on coco2017 dataset. 

## :warning: Warning
Only few models in `timm` (resnet, xcit) are tested so far. Other models might require some changes or fixes. 
