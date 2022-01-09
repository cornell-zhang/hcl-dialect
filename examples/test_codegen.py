import io
from mlir.ir import Context, Module
import hcl_mlir

if __name__ == "__main__":
    infile = open("../test/gemm.mlir","r").read()
    ctx = Context()
    mod = Module.parse(infile, ctx)
    mod.dump()
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(mod, buf)
    buf.seek(0)
    print(buf.read())