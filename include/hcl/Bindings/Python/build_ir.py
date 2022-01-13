from mlir.ir import *
from mlir.dialects import builtin, std, memref
from .affine import AffineForOp, AffineYieldOp
import mlir.ir as ir
from ._mlir_libs._hcl import *

global_ctx = Context()
global_loc = Location.unknown(global_ctx)
f32 = F32Type.get(global_ctx)
i32 = IntegerType.get_signless(32, context=global_ctx)
register_dialects(global_ctx)

global_ip = None


def get_context():
    return global_ctx


def get_location():
    return global_loc


def get_insertion_point():
    return global_ip


def set_insertion_point(ip):
    global global_ip
    global_ip = ip


def floating_point_error(op_name):
    return RuntimeError("{} does not support floating point inputs".format(op_name))


class ExprOp(object):
    def __init__(self, op):
        self.op = op

    @staticmethod
    def generic_op(OpClass, lhs, rhs):
        if isinstance(lhs.op, BlockArgument):
            expr = OpClass(i32, lhs, rhs)
            expr.op = expr.op["i"]
        else:
            expr = OpClass(f32, lhs, rhs)
            expr.op = expr.op["f"]
        return expr

    @staticmethod
    def comparison_op(lhs, rhs, arg):
        # TODO: Fix type
        if isinstance(lhs.op, BlockArgument):
            expr = CmpOp(i32, arg, lhs, rhs)
            expr.op = expr.op["i"]
        else:
            expr = CmpOp(f32, arg, lhs, rhs)
            expr.op = expr.op["f"]
        return expr

    def __add__(self, other):
        return self.generic_op(AddOp, self, other)

    def __radd__(self, other):
        return self.generic_op(AddOp, other, self)

    def __sub__(self, other):
        return self.generic_op(SubOp, self, other)

    def __rsub__(self, other):
        return self.generic_op(SubOp, other, self)

    def __mul__(self, other):
        return self.generic_op(MulOp, self, other)

    def __rmul__(self, other):
        return self.generic_op(MulOp, other, self)

    def __div__(self, other):
        return self.generic_op(DivOp, self, other)

    def __rdiv__(self, other):
        return self.generic_op(DivOp, other, self)

    def __truediv__(self, other):
        return self.generic_op(DivOp, self, other)

    def __rtruediv__(self, other):
        return self.generic_op(DivOp, other, self)

    def __floordiv__(self, other):
        return self.generic_op(FloorDivOp, self, other)

    def __rfloordiv__(self, other):
        return self.generic_op(FloorDivOp, other, self)

    def __mod__(self, other):
        return self.generic_op(RemOp, self, other)

    def __neg__(self):
        if isinstance(self.op, BlockArgument):
            op = NegOp(i32, self, loc=get_location())
        else:
            op = NegOp(f32, self, loc=get_location())
        return op

    def __lshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Left shift")
        return LeftShiftOp(self, other)

    def __rshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Right shift")
        return RightShiftOp(self, other)

    def __and__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise And")
        return AndOp(self, other)

    def __or__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise Or")
        return OrOp(self, other)

    def __xor__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise XOr")
        return XOrOp(self, other)

    def __invert__(self):
        raise RuntimeError("Not implemented")

    def __lt__(self, other):
        return self.comparison_op(self, other, arg="slt")

    def __le__(self, other):
        return self.comparison_op(self, other, arg="sle")

    def __eq__(self, other):
        return self.comparison_op(self, other, arg="eq")

    def __ne__(self, other):
        return self.comparison_op(self, other, arg="ne")

    def __gt__(self, other):
        return self.comparison_op(self, other, arg="sgt")

    def __ge__(self, other):
        return self.comparison_op(self, other, arg="sge")

    def __getitem__(self, indices):
        raise RuntimeError("Not implemented")

    def __setitem__(self, indices, expr):
        raise RuntimeError("Cannot set bit/slice of an expression")

    def __nonzero__(self):
        raise RuntimeError(
            "1) Cannot use and / or / not operator to Expr, "
            + "2) Cannot compare NumPy numbers with HeteroCL exprs, "
            + "hint: swap the operands"
        )

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
        return self.comparison_op(self, other, arg="eq")

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


class ReduceVar(IterVar):
    def __init__(self, op, bound=None, name=""):
        super(IterVar, self).__init__(op)
        self.name = name
        self.bound = bound

    def get_lower_bound(self):
        return self.bound[0]

    def get_upper_bound(self):
        return self.bound[1]


class ConstantOp(ExprOp):
    def __init__(self, dtype, val):
        super().__init__(std.ConstantOp)
        self.val = val
        self.dtype = dtype


class BinaryOp(ExprOp):
    def __init__(self, op, res_type, lhs, rhs):
        super().__init__(op)
        self.res_type = res_type
        self.lhs = lhs
        self.rhs = rhs


class CmpOp(BinaryOp):
    def __init__(self, op, res_type, lhs, rhs, arg):
        super().__init__({"f": std.CmpFOp, "i": std.CmpIOp}, res_type, lhs, rhs)
        self.arg = arg


class NegOp(ExprOp):
    def __init__(self, res_type, expr):
        super().__init__({"f": std.NegFOp, "i": std.NegFOp})  # use the same op
        self.res_type = res_type
        self.expr = expr


class AddOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__({"f": std.AddFOp, "i": std.AddIOp}, res_type, lhs, rhs)


class SubOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__({"f": std.SubFOp, "i": std.SubIOp}, res_type, lhs, rhs)


class MulOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__({"f": std.MulFOp, "i": std.MulIOp}, res_type, lhs, rhs)


class DivOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__({"f": std.DivFOp, "i": std.SignedDivIOp}, res_type, lhs, rhs)


class FloorDivOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__(
            {"f": std.SignedFloorDivIOp, "i": std.SignedFloorDivIOp},  # not supported!
            res_type,
            lhs,
            rhs,
        )


class RemOp(BinaryOp):
    def __init__(self, res_type, lhs, rhs):
        super().__init__({"f": std.RemFOp, "i": std.SignedRemIOp}, res_type, lhs, rhs)


class LeftShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.ShiftLeftOp, i32, lhs, rhs)


class RightShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.SignedShiftRightOp, i32, lhs, rhs)


class AndOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.AndOp, i32, lhs, rhs)


class OrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.OrOp, i32, lhs, rhs)


class XOrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.XOrOp, i32, lhs, rhs)


class CastOp(ExprOp):
    pass


class LoadOp(ExprOp):
    def __init__(self, res_type, tensor, indices):
        super().__init__(memref.LoadOp)
        self.res_type = res_type
        self.tensor = tensor
        self.indices = indices


class StoreOp(ExprOp):
    def __init__(self, val, to_tensor, indices):
        super().__init__(memref.StoreOp)
        self.val = val
        self.to_tensor = to_tensor
        self.indices = indices


class TensorOp(ExprOp):
    def __init__(self, shape, op, memref_type):
        super(TensorOp, self).__init__(op)
        self.shape = shape
        self.memref_type = memref_type

    def __getitem__(self, indices):
        # only one dimension
        if not isinstance(indices, tuple):
            indices = [indices]

        # format indices
        new_indices = []
        for index in indices:
            if isinstance(index, int):
                index = ConstantOp(i32, index)
            new_indices.append(index)
        return LoadOp(f32, self, new_indices)


class SumOp(ExprOp):
    def __init__(self, op, axis):
        super().__init__(op)
        self.axis = axis


class ASTBuilder:
    def visit(self, expr):
        """Apply the visitor to an expression."""

        if isinstance(expr, BinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, LoadOp):
            return self.visit_load_op(expr)
        elif isinstance(expr, SumOp):
            return self.visit_sum_op(expr)
        elif isinstance(expr, ConstantOp):
            return self.visit_constant_op(expr)
        elif isinstance(expr, (int, float)):
            return self.visit_py_builtin(expr)
        else:  # IterVar
            return expr.op  # BlockArgument

    def visit_py_builtin(self, expr):
        if isinstance(expr, int):
            cst = ConstantOp(i32, expr)
            return self.visit(cst)
        elif isinstance(expr, float):
            cst = ConstantOp(f32, expr)
            return self.visit(cst)
        else:
            raise RuntimeError("Not implemented python builtin type")

    def visit_binary_op(self, expr):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)
        if not isinstance(lhs, BlockArgument):
            lhs = lhs.result
        if not isinstance(rhs, BlockArgument):
            rhs = rhs.result
        return expr.op(expr.res_type, lhs, rhs, ip=get_insertion_point())

    def visit_load_op(self, expr):
        new_indices = []
        for index in expr.indices:
            op = self.visit(index)
            try:
                new_indices.append(op.result)
            except:
                new_indices.append(op)
        return expr.op(
            expr.res_type, expr.tensor.op.result, new_indices, ip=get_insertion_point()
        )

    def visit_constant_op(self, expr):
        if isinstance(expr.dtype, IntegerType):
            value_attr = IntegerAttr.get(IntegerType.get_signless(32), expr.val)
        elif isinstance(expr.dtype, F32Type):
            value_attr = FloatAttr.get(F32Type.get(), expr.val)
        else:
            raise RuntimeError("Type not implemented")
        return std.ConstantOp(expr.dtype, value_attr, ip=get_insertion_point())

    def visit_sum_op(self, expr):
        # save insetion point
        save_ip = get_insertion_point()

        # create a single-element register for summation
        memref_type = MemRefType.get((1,), f32)
        rv = memref.AllocOp(memref_type, None, None, None, ip=get_insertion_point())

        # create reduction loop
        if not isinstance(expr.axis, list):
            new_axes = [expr.axis]
        else:
            new_axes = expr.axis
        for axis in new_axes:
            reduction_loop = make_constant_for(
                axis.get_lower_bound(),
                axis.get_upper_bound(),
                step=1,
                name=axis.name,
                ip=get_insertion_point(),
            )
            # update insertion point
            set_insertion_point(InsertionPoint(reduction_loop.body))

            # update reduction variable
            axis.op = reduction_loop.induction_variable

        # visit subexpressions
        data = self.visit(expr.op)

        # load register value and sum up
        # value_attr should be index type, since it's an index
        value_attr = IntegerAttr.get(IndexType.get(), 0)
        zero_idx = std.ConstantOp(IndexType.get(), value_attr, ip=get_insertion_point())
        load = memref.LoadOp(
            f32, rv.result, [zero_idx.result], ip=get_insertion_point()
        )
        iter_sum = std.AddFOp(f32, data.result, load.result, ip=get_insertion_point())

        # store the result back to register
        ret_val = memref.StoreOp(
            iter_sum.result, rv.result, [zero_idx.result], ip=get_insertion_point()
        )

        # set terminator
        AffineYieldOp([], ip=get_insertion_point())

        # restore insertion point
        set_insertion_point(save_ip)
        return rv


def placeholder(shape, name=""):
    memref_type = MemRefType.get(shape, f32, loc=get_location())
    tensor = TensorOp(shape, memref.AllocOp, memref_type)
    # not sure if good or bad
    tensor.op = tensor.op(
        tensor.memref_type, None, None, None, ip=get_insertion_point()
    )
    return tensor


def reduce_axis(lb, ub, name=""):
    return ReduceVar(None, bound=(lb, ub), name=name)


def sum(expr, axis=None):
    return SumOp(expr, axis)


def make_constant_for(lb, ub, step=1, name="", stage="", reduction=False, ip=None):
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
    forOp = AffineForOp(
        None,
        None,
        step,
        lbMapAttr,
        ubMapAttr,
        name=StringAttr.get(name),
        stage=("" if stage == "" else StringAttr.get(stage)),
        reduction=(IntegerAttr.get(i32, 1) if reduction else None),
        ip=ip,
    )

    return forOp
