from mlir.ir import *
from mlir.dialects import builtin, std, memref
from .affine import AffineForOp

ctx = Context()
loc = Location.unknown(ctx)
f32 = F32Type.get(ctx)
i32 = IntegerType.get_signless(32, context=ctx)

def get_expr_op(expr, ip=None):
    if not isinstance(expr, ExprOp):
        if type(expr) == int:
            expr = IntegerAttr.get(i32, expr)
            expr = ConstantOp(std.ConstantOp(i32, expr, loc=loc, ip=ip))
        else:
            expr = FloatAttr.get(f32, expr)
            expr = ConstantOp(std.ConstantOp(f32, expr, loc=loc, ip=ip))
    return expr

def get_ip(lhs, rhs):
    if isinstance(lhs, ExprOp) and lhs.ip != None:
        return lhs.ip
    elif isinstance(rhs, ExprOp) and rhs.ip != None:
        return rhs.ip
    else:
        raise RuntimeError("Cannot find insertion point")

class ExprOp(object):

    def __init__(self, op, ip=None):
        self.op = op
        self.ip = ip

    @staticmethod
    def set_ip(expr, ip):
        if expr.ip == None:
            expr.ip = ip

    def get_expr(self):
        if isinstance(self.op, BlockArgument):
            return self.op
        elif isinstance(self.op, Attribute):
            return self.op
        else:
            return self.op.result

    @staticmethod
    def binary_op(bin_op, lhs, rhs):
        ip = get_ip(lhs, rhs)
        lhs = get_expr_op(lhs, ip)
        rhs = get_expr_op(rhs, ip)
        if isinstance(lhs.op, BlockArgument):
            op = bin_op(i32, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
        else:
            op = bin_op(f32, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
        return op, ip

    def __add__(self, other):
        return AddOp(*self.binary_op(std.AddFOp, self, other))

    def __radd__(self, other):
        return AddOp(*self.binary_op(std.AddFOp, other, self))

    def __sub__(self, other):
        return SubOp(*self.binary_op(std.SubFOp, self, other))

    def __rsub__(self, other):
        return SubOp(*self.binary_op(std.SubFOp, other, self))

    def __mul__(self, other):
        return MulOp(*self.binary_op(std.MulFOp, self, other))

    def __rmul__(self, other):
        return MulOp(*self.binary_op(std.MulFOp, other, self))

    def __div__(self, other):
        return DivOp(*self.binary_op(std.DivFOp, self, other))

    def __rdiv__(self, other):
        return DivOp(*self.binary_op(std.DivFOp, other, self))

    def __truediv__(self, other):
        return DivOp(*self.binary_op(std.DivFOp, self, other))

    def __rtruediv__(self, other):
        return DivOp(*self.binary_op(std.DivFOp, other, self))

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
    pass

class IntAttr(ExprOp):
    pass

class FloatAttr(ExprOp):
    pass

class ConstantOp(ExprOp):
    pass

class AddOp(ExprOp):
    pass

class SubOp(ExprOp):
    pass

class MulOp(ExprOp):
    pass

class DivOp(ExprOp):
    pass

class LoadOp(ExprOp):
    pass

class TensorOp(object):

    def __init__(self, op, ip):
        self.op = op
        self.ip = ip

    def __getitem__(self, indices):
        ip = indices[0].ip
        new_indices = []
        for index in indices:
            if isinstance(index, IterVar):
                new_indices.append(index.op)
            else:
                new_indices.append(index.op.result)
        return LoadOp(memref.LoadOp(f32, self.op.result, new_indices, loc=loc, ip=ip), ip)

def placeholder(shape, name="", ip=None):
    memref_type = MemRefType.get(shape, f32)
    return TensorOp(memref.AllocOp(memref_type, None, None, None, ip=ip), ip)

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