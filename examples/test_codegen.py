import io
from mlir.ir import Context, Module
import hcl_mlir

mlir_code = """
module {
    func @gemm(%A: tensor<1024x512xf32>, %B: tensor<512x1024xf32>, %C: tensor<1024x1024xf32>)
    {
        affine.for %i = 0 to 1024 {
            affine.for %j = 0 to 1024 {
                affine.for %k = 0 to 512 {
                    %a = tensor.extract %A[%i, %k] : tensor<1024x512xf32>
                    %b = tensor.extract %B[%k, %j] : tensor<512x1024xf32>
                    %c = tensor.extract %C[%i, %j] : tensor<1024x1024xf32>
                    %prod = mulf %a, %b : f32
                    %sum = addf %prod, %c: f32
                    tensor.insert %sum into %C[%i, %j] : tensor<1024x1024xf32>
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