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
            return CastOp(std.IndexCastOp(i32, self.op, ip=self.ip), self.ip).op.result # explicitly index cast
        elif isinstance(self.op, Attribute):
            return self.op
        else:
            return self.op.result

    @staticmethod
    def binary_op(bin_op, lhs, rhs, arg=""):
        ip = get_ip(lhs, rhs)
        lhs = get_expr_op(lhs, ip)
        rhs = get_expr_op(rhs, ip)
        if arg == "":
            if isinstance(lhs.op, BlockArgument):
                op = bin_op["i"](i32, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
            else:
                op = bin_op["f"](f32, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
        else:
            if isinstance(lhs.op, BlockArgument):
                op = bin_op["i"](i32, arg, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
            else:
                op = bin_op["f"](f32, arg, lhs.get_expr(), rhs.get_expr(), loc=loc, ip=ip)
        return op, ip

    def __add__(self, other):
        return AddOp(*self.binary_op({"f": std.AddFOp, "i": std.AddIOp}, self, other))

    def __radd__(self, other):
        return AddOp(*self.binary_op({"f": std.AddFOp, "i": std.AddIOp}, other, self))

    def __sub__(self, other):
        return SubOp(*self.binary_op({"f": std.SubFOp, "i": std.SubIOp}, self, other))

    def __rsub__(self, other):
        return SubOp(*self.binary_op({"f": std.SubFOp, "i": std.SubIOp}, other, self))

    def __mul__(self, other):
        return MulOp(*self.binary_op({"f": std.MulFOp, "i": std.MulIOp}, self, other))

    def __rmul__(self, other):
        return MulOp(*self.binary_op({"f": std.MulFOp, "i": std.MulIOp}, other, self))

    def __div__(self, other):
        return DivOp(*self.binary_op({"f": std.DivFOp, "i": std.DivIOp}, self, other))

    def __rdiv__(self, other):
        return DivOp(*self.binary_op({"f": std.DivFOp, "i": std.DivIOp}, other, self))

    def __truediv__(self, other):
        return DivOp(*self.binary_op({"f": std.DivFOp, "i": std.DivIOp}, self, other))

    def __rtruediv__(self, other):
        return DivOp(*self.binary_op({"f": std.DivFOp, "i": std.DivIOp}, other, self))

    def __floordiv__(self, other):
        return FloorDivOp(*self.binary_op({"f": std.SignedFloorDivFOp, "i": std.SignedFloorDivIOp}, other, self))

    def __rfloordiv__(self, other):
        return FloorDivOp(*self.binary_op({"f": std.SignedFloorDivFOp, "i": std.SignedFloorDivIOp}, other, self))

    def __mod__(self, other):
        return RemOp(*self.binary_op({"f": std.RemFOp, "i": std.RemIOp}, other, self))

    def __neg__(self):
        if isinstance(self.op, BlockArgument):
            op = NegIOp(i32, self.get_expr(), loc=loc, ip=ip)
        else:
            op = NegFOp(f32, self.get_expr(), loc=loc, ip=ip)
        return op

    def __lshift__(self, other):
        return ShiftOp(*self.binary_op({"f": std.ShiftLeftOp, "i": std.ShiftLeftOp}, self, other))

    def __rshift__(self, other):
        return ShiftOp(*self.binary_op({"f": std.SignedShiftRightOp, "i": std.SignedShiftRightOp}, other, self))

    def __and__(self, other):
        # TODO: emit error when accepting floating points
        return AndOp(*self.binary_op({"f": std.AndOp, "i": std.AndOp}, self, other))

    def __or__(self, other):
        # TODO: emit error when accepting floating points
        return OrOp(*self.binary_op({"f": std.OrOp, "i": std.OrOp}, self, other))

    def __xor__(self, other):
        # TODO: emit error when accepting floating points
        return XOrOp(*self.binary_op({"f": std.XOrOp, "i": std.XOrOp}, self, other))

    def __invert__(self):
        raise RuntimeError("Not implemented")

    def __lt__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="slt"))

    def __le__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="sle"))

    def __eq__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="eq"))

    def __ne__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="ne"))

    def __gt__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="sgt"))

    def __ge__(self, other):
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="sge"))

    def __getitem__(self, indices):
        raise RuntimeError("Not implemented")

    def __setitem__(self, indices, expr):
        raise RuntimeError("Cannot set bit/slice of an expression")

    def __nonzero__(self):
        raise RuntimeError("1) Cannot use and / or / not operator to Expr, " +
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
        return CmpOp(*self.binary_op({"f": std.CmpFOp, "i": std.CmpIOp}, self, other, arg="eq"))

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
        raise RuntimeError("Not implemented")

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

class RemOp(ExprOp):
    pass

class FloorDivOp(ExprOp):
    pass

class ShiftOp(ExprOp):
    pass

class LoadOp(ExprOp):
    pass

class StoreOp(ExprOp):
    pass

class NegOp(ExprOp):
    pass

class CmpOp(ExprOp):
    pass

class AndOp(ExprOp):
    pass

class OrOp(ExprOp):
    pass

class XOrOp(ExprOp):
    pass

class CastOp(ExprOp):
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