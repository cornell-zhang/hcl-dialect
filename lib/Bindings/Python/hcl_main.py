import mlir.ir
import io
import hcl_mlir._mlir_libs._hcl as hcl

if __name__ == "__main__":
    infile = open("../../../test/gemm.mlir","r").read()
    ctx = mlir.ir.Context()
    mod = mlir.ir.Module.parse(infile, ctx)
    mod.dump()
    buf = io.StringIO()
    hcl.emit_hlscpp(mod, buf)
    buf.seek(0)
    print(buf.read())