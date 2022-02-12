# RUN: %PYTHON %s

from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d

with Context() as ctx:
    hcl_d.register_dialect()
    print("Registration done!")

    module = Module.parse("""
        %0 = arith.constant 2 : i32
        """)
    print(str(module))
    print("Done module parsing!")

    hcl_d.loop_transformation(module)
    print("Done loop transformation!")