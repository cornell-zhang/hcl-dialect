from mlir.ir import *
from mlir.dialects import builtin, std
from .affine import AffineForOp

ctx = Context()
loc = Location.unknown(ctx)
f32 = F32Type.get(ctx)

class ExprOp(object):

    def __init__(self, ip):
        self.ip = ip

    @staticmethod
    def get_ip(lhs, rhs):
        if lhs.ip != None:
            return lhs.ip
        elif rhs.ip != None:
            return rhs.ip
        else:
            raise RuntimeError("Cannot find insertion point")

    @staticmethod
    def set_ip(expr, ip):
        if not hasattr(expr, "ip") or expr.ip == None:
            expr.ip = ip

    def get_expr(self):
        if isinstance(self.op, BlockArgument):
            return self.op
        else:
            return self.op.result

    @staticmethod
    def binary_op(bin_op, lhs, rhs):
        ip = lhs.get_ip(lhs, rhs)
        op = bin_op(f32, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
        return op

    def __add__(self, other):
        self.op = self.binary_op(std.AddFOp, self, other)
        return self

    def __radd__(self, other):
        self.op = self.binary_op(std.AddFOp, other, self)
        return self

    def __sub__(self, other):
        self.op = self.binary_op(std.SubFOp, self, other)
        return self

    def __rsub__(self, other):
        self.op = self.binary_op(std.SubFOp, other, self)
        return self

    def __mul__(self, other):
        self.op = self.binary_op(std.MulFOp, self, other)
        return self

    def __rmul__(self, other):
        self.op = self.binary_op(std.MulFOp, other, self)
        return self

    def __div__(self, other):
        self.op = self.binary_op(std.DivFOp, self, other)
        return self

    def __rdiv__(self, other):
        self.op = self.binary_op(std.DivFOp, other, self)
        return self

    def __truediv__(self, other):
        raise RuntimeError("Not implemented")

    def __rtruediv__(self, other):
        raise RuntimeError("Not implemented")

    def __floordiv__(self, other):
        raise RuntimeError("Not implemented")

    def __rfloordiv__(self, other):
        raise RuntimeError("Not implemented")

    def __mod__(self, other):
        raise RuntimeError("Not implemented")

    def __neg__(self):
        raise RuntimeError("Not implemented")
        neg_one = _api_internal._const(-1, self.dtype)
        return self.__mul__(neg_one)

    def __lshift__(self, other):
        if "float" in self.dtype:
            raise APIError("Cannot perform shift with float")
        return _make.Call(self.dtype, "shift_left", [self, other], Call.PureIntrinsic, None, 0)

    def __rshift__(self, other):
        if "float" in self.dtype:
            raise APIError("Cannot perform shift with float")
        return _make.Call(self.dtype, "shift_right", [self, other], Call.PureIntrinsic, None, 0)

    def __and__(self, other):
        if "float" in self.dtype:
            raise APIError("Cannot perform bitwise and with float")
        return _make.Call(self.dtype, "bitwise_and", [self, other], Call.PureIntrinsic, None, 0)

    def __or__(self, other):
        if "float" in self.dtype:
            raise APIError("Cannot perform bitwise or with float")
        return _make.Call(self.dtype, "bitwise_or", [self, other], Call.PureIntrinsic, None, 0)

    def __xor__(self, other):
        if "float" in self.dtype:
            raise APIError("Cannot perform bitwise xor with float")
        return _make.Call(self.dtype, "bitwise_xor", [self, other], Call.PureIntrinsic, None, 0)

    def __invert__(self):
        return _make.Call(self.dtype, "bitwise_not", [self], Call.PureIntrinsic, None, 0)

    def __lt__(self, other):
        return _make.LT(self, other)

    def __le__(self, other):
        return _make.LE(self, other)

    def __eq__(self, other):
        return EqualOp(self, other)

    def __ne__(self, other):
        return NotEqualOp(self, other)

    def __gt__(self, other):
        return _make.GT(self, other)

    def __ge__(self, other):
        return _make.GE(self, other)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            return _make.GetSlice(self, indices.start, indices.stop)
        else:
            return _make.GetBit(self, indices)

    def __setitem__(self, indices, expr):
        raise APIError("Cannot set bit/slice of an expression")

    def __nonzero__(self):
        raise APIError("1) Cannot use and / or / not operator to Expr, " +
                       "2) Cannot compare NumPy numbers with HeteroCL exprs, " +
                       "hint: swap the operands")

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other):
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : Expr
            The other expression

        Returns
        -------
        ret : Expr
            The equality expression.
        """
        return _make.EQ(self, other)

    def astype(self, dtype):
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        Returns
        -------
        expr : Expr
            Expression with new type
        """
        return _make.static_cast(dtype, self)

class IterVar(ExprOp):
    """Symbolic variable."""

    def __init__(self, val, ip):
        super(IterVar, self).__init__(ip)
        self.op = val


def make_constant_for(lb, ub, step=1, name="", ip=None):
    # Construct lower bound
    lbCst = AffineConstantExpr.get(lb)
    lbMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[lbCst])
    lbMapAttr = AffineMapAttr.get(lbMap)

    # Construct upper bound
    ubCst = AffineConstantExpr.get(ub)
    ubMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[ubCst])
    ubMapAttr = AffineMapAttr.get(ubMap)

    # Construct step
    i32 = IntegerType.get_signless(32)
    step = IntegerAttr.get(i32, step)

    # Create AffineForOp
    forOp = AffineForOp(None, None, step, lbMapAttr, ubMapAttr, name=StringAttr.get(name), ip=ip)
    return forOp