# RUN: %PYTHON %s

from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d


def test_fixed():
    with Context() as ctx, Location.unknown() as loc:
        hcl_d.register_dialect(ctx)  # need to first register dialect
        fixed_type = hcl_d.FixedType.get(12, 6)
        ufixed_type = hcl_d.UFixedType.get(20, 12)
        print(fixed_type, ufixed_type)


if __name__ == "__main__":
    test_fixed()
