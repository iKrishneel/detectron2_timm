#!/usr/bin/env python

import pytest

from timm.models import list_models

from detectron2.modeling import BACKBONE_REGISTRY

from detectron2_timm.config import get_cfg
from detectron2_timm.register import build_backbone_functions


def test_registered_size():
    in_size = len(BACKBONE_REGISTRY._obj_map)
    assert in_size == 1205  # including 3 defaults
