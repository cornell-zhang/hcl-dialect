# RUN: %PYTHON %s

import io

from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d

with Context() as ctx:
    hcl_d.register_dialect()
    print("Registration done!")

    mod = Module.parse(
        """
        func @top () -> () {
            %0 = arith.constant 2 : i32
            %1 = arith.addi %0, %0 : i32
            return
        }
        """
    )
    print(str(mod))
    print("Done module parsing!")

    hcl_d.loop_transformation(mod)
    print(str(mod))
    print("Done loop transformation!")

    buf = io.StringIO()
    hcl_d.emit_hlscpp(mod, buf)
    buf.seek(0)
    hls_code = buf.read()
    print(hls_code)
    print("Done HLS code generation")
