#!/usr/bin/env python

from typing import List

from .build import (  # NOQA: F401
    build_detectron2_backbone,
    build_detectron2_fpn_backbone,
)

from .cross_former import cross_former_s, cross_former_b  # NOQA: F401


def list_models() -> List[str]:
    return [
        'cross_former_s',
        'cross_former_b'
    ]
