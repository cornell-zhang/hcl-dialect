# RUN: %PYTHON %s

from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d

with Context() as ctx:
    hcl_d.register_dialect()
    print("Registration done!")