# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

import mlir.ir as ir
from mlir.dialects import builtin, memref, std
from mlir.ir import *

from ._mlir_libs._hcl import *
from .affine import AffineForOp, AffineYieldOp

global_ctx = Context()
global_loc = Location.unknown(global_ctx)
f32 = F32Type.get(global_ctx)
i32 = IntegerType.get_signless(32, context=global_ctx)
idx_type = IndexType.get(context=global_ctx)

register_dialects(global_ctx)


def get_context():
    return global_ctx


def get_location():
    return global_loc


def is_floating_point_type(dtype):
    return isinstance(dtype, (F16Type, F32Type, F64Type))


def is_integer_type(dtype):
    return isinstance(dtype, IntegerType)


def get_mlir_type(dtype):
    if is_integer_type(dtype) or is_floating_point_type(dtype):
        return dtype
    elif isinstance(dtype, str):
        with get_context() as ctx:
            if dtype[0:3] == "int":
                return IntegerType.get_signless(int(dtype[3:]))
            elif dtype[0:4] == "uint":
                return IntegerType.get_signless(int(dtype[4:]))
            elif dtype[0:5] == "float":
                if dtype[5:] == "16":
                    return F16Type.get()
                elif dtype[5:] == "32":
                    return F32Type.get()
                elif dtype[5:] == "64":
                    return F64Type.get()
                else:
                    raise RuntimeError("Not supported floating point type")
            elif dtype[0:5] == "fixed":
                strs = dtype[5:].split("_")
                width = IntegerAttr.get(i32, int(strs[0]))
                frac = IntegerAttr.get(i32, int(strs[1]))
                return FixedType.get(width, frac)
            elif dtype[0:6] == "ufixed":
                strs = dtype[6:].split("_")
                width = IntegerAttr.get(i32, int(strs[0]))
                frac = IntegerAttr.get(i32, int(strs[1]))
                return UFixedType.get(width, frac)
            else:
                raise RuntimeError("Unrecognized data type")
    else:
        raise RuntimeError("Unrecognized data type format")


class HCLMLIRInsertionPoint(object):
    def __init__(self):
        self.ip_stack = []

    def clear(self):
        self.ip_stack = []

    def get(self):
        return self.ip_stack[-1]

    def save(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def restore(self):
        return self.ip_stack.pop()


GlobalInsertionPoint = HCLMLIRInsertionPoint()


def floating_point_error(op_name):
    return RuntimeError("{} does not support floating point inputs".format(op_name))


def get_hcl_op(expr):
    if isinstance(expr, (int, float)):
        if isinstance(expr, int):
            return ConstantOp(i32, expr)
        elif isinstance(expr, float):
            return ConstantOp(f32, expr)
    else:
        return expr


class ExprOp(object):
    def __init__(self, op, dtype=None):
        self.op = op
        self.dtype = dtype

    @staticmethod
    def generic_op(OpClass, lhs, rhs):
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)
        if lhs.dtype != rhs.dtype:
            raise RuntimeError("Types of LHS and RHS should be the same")
        dtype = lhs.dtype
        if is_integer_type(dtype):
            expr = OpClass(dtype, lhs, rhs)
            expr.op = expr.op["i"]
        elif is_floating_point_type(dtype):
            expr = OpClass(dtype, lhs, rhs)
            expr.op = expr.op["f"]
        else:
            raise RuntimeError("Unsupported types")
        return expr

    @staticmethod
    def comparison_op(lhs, rhs, arg):
        if lhs.op.dtype != rhs.op.dtype:
            raise RuntimeError("Types of LHS and RHS should be the same")
        dtype = lhs.op.dtype
        if is_integer_type(dtype):
            expr = CmpOp(i32, arg, lhs, rhs)
            expr.op = expr.op["i"]
        elif is_floating_point_type(dtype):
            expr = CmpOp(f32, arg, lhs, rhs)
            expr.op = expr.op["f"]
        else:
            raise RuntimeError("Unsupported types")
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
            op = NegOp(self.dtype, self, loc=get_location())
        else:
            op = NegOp(self.dtype, self, loc=get_location())
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


#################################################
#
# AST leaf nodes
#
#################################################


class IterVar(ExprOp):
    """loop induction variable (BlockArgument)"""

    def __init__(self, op):
        super().__init__(op, dtype=i32)


class ReduceVar(IterVar):
    """reduce_axis
    induction variable of reduction loop
    """

    def __init__(self, op, bound=None, name=""):
        super().__init__(op)
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
        self.dtype = get_mlir_type(dtype)


class TensorOp(ExprOp):
    def __init__(self, shape, op, dtype, name=None):
        super().__init__(op)
        self.shape = shape
        self.dtype = get_mlir_type(dtype)
        self.name = name

    def build(self):
        memref_type = self.get_memref_type()
        self.op = self.op(memref_type, None, None, None, ip=GlobalInsertionPoint.get())
        return self.op

    def get_memref_type(self):
        return MemRefType.get(self.shape, self.dtype, loc=get_location())

    def set_axis(self, _axis):
        self._axis = _axis

    @property
    def axis(self):
        return self._axis

    def __getitem__(self, indices):
        # only one dimension
        if not isinstance(indices, tuple):
            indices = [indices]

        # format indices
        new_indices = []
        for index in indices:
            if isinstance(index, int):
                index = ConstantOp(idx_type, index)
            new_indices.append(index)
        return LoadOp(self.dtype, self, new_indices)


#################################################
#
# AST inner nodes
#
#################################################


class BinaryOp(ExprOp):
    def __init__(self, op, dtype, lhs, rhs):
        super().__init__(op)
        self.dtype = dtype
        self.lhs = lhs
        self.rhs = rhs


class CmpOp(BinaryOp):
    def __init__(self, op, dtype, lhs, rhs, arg):
        super().__init__({"f": std.CmpFOp, "i": std.CmpIOp}, dtype, lhs, rhs)
        self.arg = arg


class NegOp(ExprOp):
    def __init__(self, dtype, expr):
        super().__init__({"f": std.NegFOp, "i": std.NegFOp})  # use the same op
        self.dtype = dtype
        self.expr = expr


class AddOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"f": std.AddFOp, "i": std.AddIOp}, dtype, lhs, rhs)


class SubOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"f": std.SubFOp, "i": std.SubIOp}, dtype, lhs, rhs)


class MulOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"f": std.MulFOp, "i": std.MulIOp}, dtype, lhs, rhs)


class DivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"f": std.DivFOp, "i": std.SignedDivIOp}, dtype, lhs, rhs)


class FloorDivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"f": std.SignedFloorDivIOp, "i": std.SignedFloorDivIOp},  # not supported!
            dtype,
            lhs,
            rhs,
        )


class RemOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"f": std.RemFOp, "i": std.SignedRemIOp}, dtype, lhs, rhs)


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
    def __init__(self, dtype, tensor, indices):
        super().__init__(memref.LoadOp, dtype)
        self.tensor = tensor
        self.indices = indices


class StoreOp(ExprOp):
    def __init__(self, val, to_tensor, indices):
        super().__init__(memref.StoreOp)
        self.val = val
        self.to_tensor = to_tensor
        self.indices = indices


class SumOp(ExprOp):
    def __init__(self, op, axis, dtype):
        super().__init__(op, dtype=get_mlir_type(dtype))
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
        else:  # IterVar
            return expr.op  # BlockArgument

    def visit_binary_op(self, expr):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)
        if not isinstance(lhs, BlockArgument):
            lhs = lhs.result
        if not isinstance(rhs, BlockArgument):
            rhs = rhs.result
        return expr.op(expr.dtype, lhs, rhs, ip=GlobalInsertionPoint.get())

    def visit_load_op(self, expr):
        new_indices = []
        for index in expr.indices:
            op = self.visit(index)
            try:
                new_indices.append(op.result)
            except:
                new_indices.append(op)
        try:
            tensor = expr.tensor.op.result
        except:  # BlockArgument
            tensor = expr.tensor.op
        return expr.op(expr.dtype, tensor, new_indices, ip=GlobalInsertionPoint.get())

    def visit_constant_op(self, expr):
        if isinstance(expr.dtype, IntegerType):
            value_attr = IntegerAttr.get(i32, expr.val)
        elif isinstance(expr.dtype, F32Type):
            value_attr = FloatAttr.get(f32, expr.val)
        else:
            raise RuntimeError("Type not implemented")
        return std.ConstantOp(expr.dtype, value_attr, ip=GlobalInsertionPoint.get())

    def visit_sum_op(self, expr):
        # save insetion point
        save_ip = GlobalInsertionPoint.get()

        # create a single-element register for summation
        dtype = expr.dtype
        memref_type = MemRefType.get((1,), dtype)
        rv = memref.AllocOp(
            memref_type, None, None, None, ip=GlobalInsertionPoint.get()
        )

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
                ip=GlobalInsertionPoint.get(),
            )
            # update insertion point
            GlobalInsertionPoint.save(reduction_loop.body)

            # update reduction variable
            axis.op = reduction_loop.induction_variable

        # visit subexpressions
        data = self.visit(expr.op)

        # load register value and sum up
        zero_idx = std.ConstantOp(
            idx_type, IntegerAttr.get(idx_type, 0), ip=GlobalInsertionPoint.get()
        )
        load = memref.LoadOp(
            dtype, rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        if is_floating_point_type(dtype):
            add_op = std.AddFOp
        elif is_integer_type(dtype):
            add_op = std.AddIOp
        else:
            raise RuntimeError("Unsupported type")
        if dtype != data.result.type:
            raise RuntimeError(
                "Reduction variable should have the same type with the data. Got {} and {}".format(
                    dtype, data.result.type
                )
            )
        iter_sum = add_op(
            dtype, data.result, load.result, ip=GlobalInsertionPoint.get()
        )

        # store the result back to register
        memref.StoreOp(
            iter_sum.result, rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )

        # set terminator
        AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # restore insertion point
        GlobalInsertionPoint.restore()
        return rv


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
