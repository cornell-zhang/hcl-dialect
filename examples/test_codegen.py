import io
from mlir.ir import Context, Module
import hcl_mlir

mlir_code = """
module {
    func @gemm(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
    {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                    %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                    %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                } { loop_name = "k" }
            } { loop_name = "j" }
        } { loop_name = "i", stage_name = "s" }
        return
    }
}
"""

if __name__ == "__main__":
    # infile = open("gemm.mlir","r").read()
    infile = mlir_code
    ctx = Context()
    mod = Module.parse(infile, ctx)
    mod.dump()
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(mod, buf)
    buf.seek(0)
    print(buf.read())