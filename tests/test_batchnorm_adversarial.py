"""Stable collection facade for decomposed BatchNorm robustness cases.

Area-specific modules own the test bodies; imports keep every historical pytest
node ID under tests/test_batchnorm_adversarial.py.
"""

from _batchnorm_fusion_structure import *
from _batchnorm_schema_cases import *
from _batchnorm_runtime_bindings import *
from _batchnorm_copy_dis_builtins import *
from _batchnorm_bootstrap_runtime import *
from _batchnorm_internal_component import *
from _batchnorm_semantic_boundary import *
