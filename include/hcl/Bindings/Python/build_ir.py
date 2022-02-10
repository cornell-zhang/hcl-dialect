# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
from contextvars import ContextVar

from mlir.dialects import builtin, math, memref, std
from mlir.ir import *

from . import affine
from ._hcl_ops_gen import *
from ._mlir_libs._hcl import *

global_ctx = Context()
global_loc = Location.unknown(global_ctx)
f32 = F32Type.get(global_ctx)
f64 = F64Type.get(global_ctx)
bool = IntegerType.get_signless(1, context=global_ctx)
i32 = IntegerType.get_signless(32, context=global_ctx)
i64 = IntegerType.get_signless(64, context=global_ctx)
idx_type = IndexType.get(context=global_ctx)

ImperativeLoopNestCount = ContextVar("ImperativeLoopNestCount", default=1)
ImperativeLoopDepth = ContextVar("ImperativeLoopDepth", default=0)
StageName = ContextVar("StageName", default="")

register_dialects(global_ctx)


class HCLFlags(object):
    def __init__(self):
        self.BUILD_INPLACE = False
        self.EXTRACT_FUNCTION = False

    def enable_build_inplace(self):
        self.BUILD_INPLACE = True

    def disable_build_inplace(self):
        self.BUILD_INPLACE = False


flags = HCLFlags()
enable_build_inplace = flags.enable_build_inplace
disable_build_inplace = flags.disable_build_inplace
EXTRACT_FUNCTION = flags.EXTRACT_FUNCTION


class UniqueName(object):
    scalar_idx = 0
    tensor_idx = 0
    stage_idx = 0

    def __init__(self):
        pass

    @classmethod
    def get(cls, case="stage"):
        if case == "stage":
            # Imperative computing stage
            name = "stage_" + str(cls.stage_idx)
            cls.stage_idx += 1
        elif case == "scalar":
            name = "scalar_" + str(cls.scalar_idx)
            cls.scalar_idx += 1
        elif case == "tensor":
            name = "compute_" + str(cls.tensor_idx)
            cls.tensor_idx += 1
        else:
            raise RuntimeError(f"Unrecognized case in get_unique_name: {case}")
        return name


def get_context():
    return global_ctx


def get_location():
    return global_loc


def is_floating_point_type(dtype):
    return isinstance(dtype, (F16Type, F32Type, F64Type))


def is_integer_type(dtype):
    return isinstance(dtype, IntegerType)


def is_fixed_type(dtype):
    return isinstance(dtype, (FixedType, UFixedType))


def is_index_type(dtype):
    return isinstance(dtype, IndexType)


def is_hcl_mlir_type(dtype):
    return (
        is_floating_point_type(dtype) or is_integer_type(dtype) or is_fixed_type(dtype)
    )


def get_mlir_type(dtype):
    if (
        is_integer_type(dtype)
        or is_floating_point_type(dtype)
        or is_fixed_type(dtype)
        or is_index_type(dtype)
    ):
        return dtype
    elif isinstance(dtype, str):
        with get_context() as ctx:
            if dtype[0:3] == "int":
                return IntegerType.get_signless(int(dtype[3:]))
            elif dtype[0:4] == "uint":
                # TODO: Support signedness
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
                return FixedType.get(int(strs[0]), int(strs[1]), ctx)
            elif dtype[0:6] == "ufixed":
                strs = dtype[6:].split("_")
                return UFixedType.get(int(strs[0]), int(strs[1]), ctx)
            else:
                raise RuntimeError("Unrecognized data type")
    else:
        raise RuntimeError("Unrecognized data type format")


def print_mlir_type(dtype):
    if is_floating_point_type(dtype):
        if dtype.width == 32:
            return "float"
        elif dtype.width == 64:
            return "double"
        else:
            raise RuntimeError("Not supported data type")
    elif is_integer_type(dtype):
        # TODO: Support signedness
        if dtype.width == 32:
            return "int"
        elif dtype.width == 64:
            return "long int"
        elif dtype.width == 1:
            return "bool"
        else:
            return "ap_int<{}>".format(dtype.width)
    elif is_fixed_type(dtype):
        if isinstance(dtype, FixedType):
            return "ap_fixed<{}, {}>".format(dtype.width, dtype.frac)
        elif isinstance(dtype, UFixedType):
            return "ap_ufixed<{}, {}>".format(dtype.width, dtype.frac)
        else:
            raise RuntimeError("Not supported data type")
    else:
        raise RuntimeError("Not supported data type")


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


def get_type_rank(dtype):
    if is_integer_type(dtype):
        base = 0
        width = dtype.width
        if width > 64:
            raise RuntimeError("Cannot support integer width larger than 64")
        base += width
        return base
    elif is_index_type(dtype):  # width 32
        base = 65
        return base
    elif is_fixed_type(dtype):
        base = 100
        width = dtype.width
        frac = dtype.frac
        base += width
        return base
    elif is_floating_point_type(dtype):
        base = 1000
        if isinstance(dtype, F16Type):
            base += 1
        elif isinstance(dtype, F32Type):
            base += 2
        elif isinstance(dtype, F64Type):
            base += 3
        else:
            raise RuntimeError("Type error: {}".format(dtype))
        return base
    else:
        raise RuntimeError("Type error: {}".format(dtype))


def cast_types(lhs, rhs):
    """
    Implementation based on
    https://en.cppreference.com/w/c/language/conversion
    """
    ltype = lhs.dtype
    rtype = rhs.dtype
    # 1) If one operand is long double (omitted)
    # 2) Otherwise, if one operand is double
    if isinstance(ltype, F64Type):
        # integer or real floating type to double
        res_type = f64
    # 3) Otherwise, if one operand is float
    elif isinstance(ltype, F32Type):
        # integer type to float (the only real type possible is float, which remains as-is)
        res_type = f32
    # 4) Otherwise, both operands are integers. Both operands undergo integer promotions (see below); then, after integer promotion, one of the following cases applies:
    elif isinstance(ltype, (IntegerType, IndexType)):
        res_type = ltype
    # 5) Fixed types
    elif is_fixed_type(ltype):
        # TODO: UFixed type
        if is_integer_type(rtype):
            res_type = FixedType.get(rtype.width, ltype.frac, global_ctx)
        else:
            raise RuntimeError("Type conversion not implemented")
    else:
        raise RuntimeError("Type conversion failed")
    print(
        "Warning: Types of {} ({}) and {} ({}) are different. Implicitly cast {} to {}.".format(
            lhs, ltype, rhs, rtype, rtype, res_type
        )
    )
    return CastOp(rhs, res_type)


def regularize_fixed_type(lhs, rhs):
    if not is_fixed_type(lhs.dtype) or not is_fixed_type(rhs.dtype):
        raise RuntimeError("Should be all fixed types")
    if not lhs.dtype.frac == rhs.dtype.frac:
        raise RuntimeError("Should have the same frac")
    lwidth = lhs.dtype.width
    rwidth = rhs.dtype.width
    if lwidth < rwidth:
        res_type = FixedType.get(rwidth, lhs.dtype.frac, global_ctx)
        cast = CastOp(lhs, res_type)
        return cast, rhs
    elif lwidth > rwidth:
        res_type = FixedType.get(lwidth, rhs.dtype.frac, global_ctx)
        cast = CastOp(rhs, res_type)
        return lhs, cast
    else:
        return lhs, rhs


class ExprOp(object):
    def __init__(self, op, dtype=None):
        self.op = op
        self.dtype = dtype
        self.built_op = None

    @property
    def result(self):
        if isinstance(self.op, BlockArgument):
            return self.op
        else:
            return self.built_op.result

    @staticmethod
    def generic_op(OpClass, lhs, rhs, arg=None):
        # turn py builtin op to hcl op
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)

        # type checking & conversion
        # get_type_rank has the valid type checking
        lrank = get_type_rank(lhs.dtype)
        rrank = get_type_rank(rhs.dtype)
        # types are the same, no need to cast
        if lrank == rrank:
            pass
        # always ensure the first op has higher ranking
        elif lrank < rrank:
            lhs = cast_types(rhs, lhs)
        else:  # lrank > rrank
            rhs = cast_types(lhs, rhs)
        if is_fixed_type(lhs.dtype) or is_fixed_type(rhs.dtype):
            lhs, rhs = regularize_fixed_type(lhs, rhs)

        # create AST node based on different types
        dtype = lhs.dtype
        if arg == None:
            expr = OpClass(dtype, lhs, rhs)
        else:
            expr = OpClass(lhs, rhs, arg)
        return expr

    @staticmethod
    def generic_integer_op(OpClass, lhs, rhs):
        # turn py builtin op to hcl op
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)
        expr = OpClass(lhs, rhs)
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
        expr = NegOp(self.dtype, self)
        return expr

    def __lshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Left shift")
        return self.generic_integer_op(LeftShiftOp, self, other)

    def __rshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Right shift")
        return self.generic_integer_op(RightShiftOp, self, other)

    def __and__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise And")
        return self.generic_integer_op(AndOp, self, other)

    def __or__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise Or")
        return self.generic_integer_op(OrOp, self, other)

    def __xor__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise XOr")
        return self.generic_integer_op(XOrOp, self, other)

    def __invert__(self):
        raise RuntimeError("Not implemented")

    def __lt__(self, other):
        return self.generic_op(CmpOp, self, other, arg="slt")

    def __le__(self, other):
        return self.generic_op(CmpOp, self, other, arg="sle")

    def __eq__(self, other):
        return self.generic_op(CmpOp, self, other, arg="eq")

    def __ne__(self, other):
        return self.generic_op(CmpOp, self, other, arg="ne")

    def __gt__(self, other):
        return self.generic_op(CmpOp, self, other, arg="sgt")

    def __ge__(self, other):
        return self.generic_op(CmpOp, self, other, arg="sge")

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
        return self.generic_op(self, other, arg="eq")

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
        super().__init__(op, dtype=idx_type)
        self.built_op = op

    def update_op(self, op):
        self.op = op
        self.built_op = op


class ReduceVar(IterVar):
    """reduce_axis
    induction variable of reduction loop
    """

    def __init__(self, op, bound=None, name=""):
        super().__init__(op)
        self.name = name
        self.bound = bound

    @property
    def lower_bound(self):
        return self.bound[0]

    @property
    def upper_bound(self):
        return self.bound[1]


class ConstantOp(ExprOp):
    def __init__(self, dtype, val):
        super().__init__(std.ConstantOp)
        self.val = val
        self.dtype = get_mlir_type(dtype)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if isinstance(self.dtype, IntegerType):
            value_attr = IntegerAttr.get(i32, self.val)
        elif isinstance(self.dtype, F32Type):
            value_attr = FloatAttr.get(f32, self.val)
        elif isinstance(self.dtype, IndexType):
            value_attr = IntegerAttr.get(idx_type, self.val)
        else:
            raise RuntimeError("Type error")
        self.built_op = self.op(self.dtype, value_attr, ip=GlobalInsertionPoint.get())
        return self.built_op


class TensorSlice(ExprOp):
    def __init__(self, full_shape, op, dtype, indices, name=None):
        super().__init__(op)
        self.op = op
        self.full_shape = full_shape
        self.dtype = dtype
        self.name = name
        self.indices = indices
        # calculate tensor slice shape
        shape = list()
        dims = 0
        for index in indices:
            if isinstance(index, int):
                dims += 1
            elif isinstance(index, slice):
                step = index.step if index.step is not None else 1
                dim_size = (index.stop - index.start) / step
                shape.append(int(dim_size))
                dims += 1
        for i, dim in enumerate(self.full_shape):
            if i < dims:
                continue
            shape.append(dim)
        self.shape = tuple(shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(self.indices + indices) < len(self.full_shape):
            return TensorSlice(
                self.full_shape, self.op, self.dtype, self.indices + indices, self.name
            )
        elif len(self.indices + indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                if isinstance(index, int):
                    index = ConstantOp(idx_type, index)
                new_indices.append(index)
            load = LoadOp(self.dtype, self, new_indices)
            # TODO(Niansong): Why build in place result in duplicate load?
            # if flags.BUILD_INPLACE:
            #     load.build()
            return load
        else:
            raise RuntimeError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(self.indices + indices) < len(self.full_shape):
            pass
        elif len(self.indices + indices) == len(self.shape):
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(idx_type, index)
                new_indices.append(index)
            return StoreOp(expr, self, self.indices + new_indices)
        else:
            raise RuntimeError("Indices length > # of array dimensions")


class TensorOp(ExprOp):
    def __init__(self, shape, op, dtype, name=None):
        # op can be BlockArgument or AllocOp.result
        super().__init__(op)
        self.shape = shape
        if isinstance(dtype, str):
            self.dtype = get_mlir_type(dtype)
        else:
            self.dtype = dtype
        self.name = name

    def build(self):
        memref_type = self.get_memref_type()
        self.built_op = self.op(
            memref_type, None, None, None, ip=GlobalInsertionPoint.get()
        )
        self.built_op.attributes["name"] = StringAttr.get(self.name)
        return self.built_op

    def get_memref_type(self):
        return MemRefType.get(self.shape, self.dtype, loc=get_location())

    def set_axis(self, _axis):
        self._axis = _axis

    @property
    def axis(self):
        return self._axis

    @property
    def v(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self.__getitem__(0)
        else:
            raise RuntimeError(".v can only be used on mutable scalars")

    @v.setter
    def v(self, value):
        """A syntactic sugar for setting the value of a single-element tensor.
        This is the same as using `a[0]=value`, where a is a single-element tensor.
        Parameters
        ----------
        value : Expr
            The value to be set
        """
        self.__setitem__(0, value)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # if we are slicing tensor
        if len(indices) < len(self.shape):
            return TensorSlice(self.shape, self.op, self.dtype, indices, self.name)
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(idx_type, index)
                new_indices.append(index)
            load = LoadOp(self.dtype, self, new_indices)
            # if flags.BUILD_INPLACE:
            #     load.build()
            return load
        else:
            raise RuntimeError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) < len(self.shape):
            # TODO(Niansong): I think this is doable actually
            raise RuntimeError("Writing to a slice of tensor is not allowed.")
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(idx_type, index)
                new_indices.append(index)
            return StoreOp(expr, self, new_indices)
        else:
            raise RuntimeError("Indices length > # of array dimensions")


#################################################
#
# AST inner nodes
#
#################################################


class UnaryOp(ExprOp):
    def __init__(self, op, dtype, val):
        super().__init__(op)
        self.dtype = dtype
        self.val = val
        if isinstance(op, dict):
            if is_integer_type(dtype):
                self.op = op["int"]
            elif is_floating_point_type(dtype):
                self.op = op["float"]
            elif is_fixed_type(dtype):
                self.op = op["fixed"]
            else:
                raise RuntimeError("Unsupported types")
        else:
            self.op = op
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
        )
        return self.built_op


class BinaryOp(ExprOp):
    def __init__(self, op, dtype, lhs, rhs):
        super().__init__(op)
        self.dtype = dtype
        self.lhs = lhs
        self.rhs = rhs
        if isinstance(op, dict):
            if is_integer_type(dtype) or is_index_type(dtype):
                self.op = op["int"]
            elif is_floating_point_type(dtype):
                self.op = op["float"]
            elif is_fixed_type(dtype):
                self.op = op["fixed"]
            else:
                raise RuntimeError("Unsupported types")
        else:
            self.op = op
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype, self.lhs.result, self.rhs.result, ip=GlobalInsertionPoint.get()
        )
        return self.built_op


class CmpOp(BinaryOp):
    def __init__(self, lhs, rhs, arg):
        self.arg = arg
        super().__init__({"float": std.CmpFOp, "int": std.CmpIOp}, bool, lhs, rhs)

    def build(self):
        if self.arg == "eq":
            attr = 0
        elif self.arg == "ne":
            attr = 1
        elif self.arg == "slt":
            attr = 2
        elif self.arg == "sle":
            attr = 3
        elif self.arg == "sgt":
            attr = 4
        elif self.arg == "sge":
            attr = 5
        elif self.arg == "ult":
            attr = 6
        elif self.arg == "ule":
            attr = 7
        elif self.arg == "ugt":
            attr = 8
        elif self.arg == "uge":
            attr = 9
        self.built_op = self.op(
            self.dtype,
            IntegerAttr.get(i64, attr),
            self.lhs.result,
            self.rhs.result,
            ip=GlobalInsertionPoint.get(),
        )
        return self.built_op


class AddOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": std.AddFOp, "int": std.AddIOp, "fixed": AddFixedOp},
            dtype,
            lhs,
            rhs,
        )


class SubOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": std.SubFOp, "int": std.SubIOp, "fixed": SubFixedOp},
            dtype,
            lhs,
            rhs,
        )


class MulOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": std.MulFOp, "int": std.MulIOp, "fixed": MulFixedOp},
            dtype,
            lhs,
            rhs,
        )


class DivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": std.DivFOp, "int": std.SignedDivIOp}, dtype, lhs, rhs
        )


class FloorDivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {
                "float": std.SignedFloorDivIOp,
                "int": std.SignedFloorDivIOp,
            },  # not supported!
            dtype,
            lhs,
            rhs,
        )


class RemOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": std.RemFOp, "int": std.SignedRemIOp}, dtype, lhs, rhs
        )


class LeftShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.ShiftLeftOp, i32, lhs, rhs)


class RightShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.SignedShiftRightOp, i32, lhs, rhs)


class AndOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.AndOp, bool, lhs, rhs)


class OrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.OrOp, bool, lhs, rhs)


class XOrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(std.XOrOp, bool, lhs, rhs)


class NegOp(UnaryOp):
    def __init__(self, dtype, val):
        super().__init__(
            {"float": std.NegFOp, "int": std.NegFOp}, dtype, val
        )  # use the same op


class MathExpOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.ExpOp, f32, val)


class MathLogOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.LogOp, f32, val)


class MathLog2Op(UnaryOp):
    def __init__(self, val):
        super().__init__(math.Log2Op, f32, val)


class MathLog10Op(UnaryOp):
    def __init__(self, val):
        super().__init__(math.Log10Op, f32, val)


class MathSqrtOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.SqrtOp, f32, val)


class MathSinOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.SinOp, f32, val)


class MathCosOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.CosOp, f32, val)


class MathTanhOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.TanhOp, f32, val)


class CastOp(ExprOp):
    def __init__(self, val, res_type=None):
        # dtype is the result type
        res_type = get_mlir_type(res_type)
        self.val = get_hcl_op(val)
        if is_index_type(res_type) and is_integer_type(self.val.dtype):
            op = std.IndexCastOp
        else:
            op = builtin.UnrealizedConversionCastOp
        super().__init__(op, res_type)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if self.op == std.IndexCastOp:
            self.built_op = self.op(
                self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
            )
        else:  # builtin.UnrealizedConversionCastOp
            self.built_op = self.op(
                [self.dtype], [self.val.result], ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class GetBitOp(ExprOp):
    def __init__(self, val, index):
        super().__init__(GetIntBitOp, bool)
        self.val = val
        self.index = index
        if not isinstance(self.index.dtype, IndexType):
            print(
                "Warning: GetBitOp's input is not an index. Cast from {} to {}.".format(
                    self.index.dtype, idx_type
                )
            )
            self.index = CastOp(self.index, idx_type)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype,
            self.val.result,
            self.index.result,
            ip=GlobalInsertionPoint.get(),
        )
        return self.built_op


class LoadOp(ExprOp):
    def __init__(self, dtype, tensor, indices):
        super().__init__(affine.AffineLoadOp, dtype)
        self.tensor = tensor
        self.indices = indices
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        new_indices = []
        for index in self.indices:
            new_indices.append(index.result)
        self.built_op = self.op(
            self.dtype, self.tensor.result, new_indices, ip=GlobalInsertionPoint.get()
        )
        self.built_op.attributes["from"] = StringAttr.get(self.tensor.name)
        return self.built_op

    def build_affine(self, indices, affine_attr):
        self.built_op = self.op(
            self.dtype,
            self.tensor.result,
            indices,
            affine_attr,
            ip=GlobalInsertionPoint.get(),
        )
        self.built_op.attributes["from"] = StringAttr.get(self.tensor.name)
        return self.built_op

    def __getitem__(self, indices):
        if not is_integer_type(self.dtype):
            raise RuntimeError("Only fixed integers can access the bits")
        if not isinstance(indices, tuple):
            indices = (indices,)
        if not len(indices) == 1:
            raise RuntimeError("Can only access one bit of the integer")
        index = indices[0]
        # TODO: Not necessary be a constant
        # if index >= self.dtype.width:
        #     raise RuntimeError(
        #         "Index ({}) is larger than the width of the integer ({})".format(
        #             index, self.dtype.width
        #         )
        #     )
        return GetBitOp(self, index)


class StoreOp(ExprOp):
    def __init__(self, val, to_tensor, indices):
        super().__init__(affine.AffineStoreOp)
        if val.dtype != to_tensor.dtype:
            print(
                "Warning: store operation has different input types. Cast from {} to {}.".format(
                    val.dtype, to_tensor.dtype
                )
            )
            val = CastOp(val, to_tensor.dtype)
        self.val = val
        self.to_tensor = to_tensor
        self.indices = indices
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        new_indices = []
        for index in self.indices:
            new_indices.append(index.result)
        self.built_op = self.op(
            self.val.result,
            self.to_tensor.result,
            new_indices,
            ip=GlobalInsertionPoint.get(),
        )
        self.built_op.attributes["to"] = StringAttr.get(self.to_tensor.name)
        return self.built_op

    def build_affine(self, indices, affine_attr):
        self.built_op = self.op(
            self.dtype,
            self.val.result,
            self.to_tensor.result,
            indices,
            affine_attr,
            ip=GlobalInsertionPoint.get(),
        )
        self.built_op.attributes["to"] = StringAttr.get(self.to_tensor.name)
        return self.built_op


class CallOp(ExprOp):
    def __init__(self, dtype, func_name, inputs):
        # here we only accept one result
        super().__init__(std.CallOp, dtype)
        self.func_name = func_name
        self.inputs = inputs
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            [self.dtype] if self.dtype != None else [],
            FlatSymbolRefAttr.get(self.func_name),
            self.inputs,
            ip=GlobalInsertionPoint.get(),
        )
        return self.built_op


class SelectOp(ExprOp):
    """Ternary operation"""

    def __init__(self, cond, true_val, false_val):
        super().__init__(std.SelectOp)
        # turn py builtin op to hcl op
        true_val = get_hcl_op(true_val)
        false_val = get_hcl_op(false_val)
        # do the testing
        if true_val.dtype != false_val.dtype:
            raise RuntimeError(
                "SelectOp should have two same type of inputs. Got {} and {}".format(
                    true_val.dtype, false_val.dtype
                )
            )
        self.dtype = true_val.dtype
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype,
            self.cond.result,
            self.true_val.result,
            self.false_val.result,
            ip=GlobalInsertionPoint.get(),
        )
        return self.built_op


class SumOp(ExprOp):
    # cannot build inplace!!!
    def __init__(self, op, axis, dtype):
        super().__init__(op, dtype=get_mlir_type(dtype))
        self.axis = axis


class ASTBuilder:
    def __init__(self):
        self.iv = []

    def visit(self, expr):
        """Apply the visitor to an expression."""

        if isinstance(expr, UnaryOp):
            return self.visit_unary_op(expr)
        elif isinstance(expr, BinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, SelectOp):
            return self.visit_ternary_op(expr)
        elif isinstance(expr, LoadOp):
            return self.visit_load_op(expr)
        elif isinstance(expr, StoreOp):
            return self.visit_store_op(expr)
        elif isinstance(expr, GetBitOp):
            return self.visit_getbit_op(expr)
        elif isinstance(expr, CastOp):
            return self.visit_cast_op(expr)
        elif isinstance(expr, SumOp):
            return self.visit_sum_op(expr)
        elif isinstance(expr, ConstantOp):
            return self.visit_constant_op(expr)
        else:  # IterVar
            return expr.built_op  # BlockArgument

    def visit_affine_expr(self, expr):
        """Build affine expression.
        * Should all be binary op
        * AffineExpr can be automatically simplied
        """
        if isinstance(expr, IterVar):
            self.iv.append(expr.op)  # BlockArgument
            return AffineExpr.get_dim(len(self.iv) - 1)
        elif isinstance(expr, ConstantOp):
            return AffineExpr.get_constant(expr.val)
        elif isinstance(expr, CastOp):
            return self.visit_affine_expr(expr.val)
        lhs = self.visit_affine_expr(expr.lhs)
        rhs = self.visit_affine_expr(expr.rhs)
        if isinstance(expr, AddOp):
            return lhs + rhs
        elif isinstance(expr, SubOp):
            return lhs - rhs
        elif isinstance(expr, MulOp):
            return lhs * rhs
        elif isinstance(expr, DivOp):
            return lhs / rhs
        elif isinstance(expr, RemOp):
            return lhs % rhs
        else:
            raise RuntimeError("Not an affine index!")

    def visit_unary_op(self, expr):
        self.visit(expr.val)
        return expr.build()

    def visit_binary_op(self, expr):
        self.visit(expr.lhs)
        self.visit(expr.rhs)
        return expr.build()

    def visit_ternary_op(self, expr):
        self.visit(expr.cond)
        self.visit(expr.true_val)
        self.visit(expr.false_val)
        return expr.build()

    def visit_load_op(self, expr):
        # create affine expressions
        self.iv = []  # clear
        exprs = []
        for index in expr.indices:
            affine_expr = self.visit_affine_expr(index)
            exprs.append(affine_expr)
        affine_map = AffineMap.get(dim_count=len(self.iv), symbol_count=0, exprs=exprs)
        affine_attr = AffineMapAttr.get(affine_map)
        return expr.build_affine(self.iv, affine_attr)

    def visit_store_op(self, expr):
        # create affine expressions
        self.iv = []  # clear
        exprs = []
        for index in expr.indices:
            affine_expr = self.visit_affine_expr(index)
            exprs.append(affine_expr)
        affine_map = AffineMap.get(dim_count=len(self.iv), symbol_count=0, exprs=exprs)
        affine_attr = AffineMapAttr.get(affine_map)
        return expr.build_affine(self.iv, affine_attr)

    def visit_cast_op(self, expr):
        self.visit(expr.val)
        return expr.build()

    def visit_getbit_op(self, expr):
        self.visit(expr.val)
        self.visit(expr.index)
        return expr.build()

    def visit_constant_op(self, expr):
        return expr.build()

    def visit_sum_op(self, expr):
        # save insetion point
        save_ip = GlobalInsertionPoint.get()

        # create a single-element register for summation
        dtype = expr.dtype
        memref_type = MemRefType.get((1,), dtype)
        rv = memref.AllocOp(
            memref_type, None, None, None, ip=GlobalInsertionPoint.get()
        )
        rv.attributes["name"] = StringAttr.get("sum_rv")
        # intialize the single-element register to zero
        zero_idx = std.ConstantOp(
            idx_type, IntegerAttr.get(idx_type, 0), ip=GlobalInsertionPoint.get()
        )
        if is_floating_point_type(dtype):
            zero_value = std.ConstantOp(
                dtype, FloatAttr.get(dtype, 0), ip=GlobalInsertionPoint.get()
            )
        elif is_integer_type(dtype) or is_fixed_type(dtype):
            zero_value = std.ConstantOp(
                dtype, IntegerAttr.get(dtype, 0), ip=GlobalInsertionPoint.get()
            )
        else:
            raise RuntimeError("Unrecognized data type in reduction sum op")

        store = affine.AffineStoreOp(
            zero_value.result,
            rv.result,
            [zero_idx.result],
            ip=GlobalInsertionPoint.get(),
        )
        store.attributes["to"] = StringAttr.get("sum_rv")

        # create reduction loop
        if not isinstance(expr.axis, list):
            new_axes = [expr.axis]
        else:
            new_axes = expr.axis
        for axis in new_axes:
            reduction_loop = make_affine_for(
                axis.lower_bound,
                axis.upper_bound,
                step=1,
                name=axis.name,
                ip=GlobalInsertionPoint.get(),
            )
            # update insertion point
            GlobalInsertionPoint.save(reduction_loop.body)

            # update reduction variable
            axis.update_op(reduction_loop.induction_variable)

        # visit subexpressions
        data = self.visit(expr.op)

        # load register value and sum up
        load = affine.AffineLoadOp(
            dtype, rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        load.attributes["from"] = StringAttr.get("sum_rv")
        if is_floating_point_type(dtype):
            add_op = std.AddFOp
        elif is_integer_type(dtype):
            add_op = std.AddIOp
        elif is_fixed_type(dtype):
            add_op = AddFixedOp
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
        store_reg = affine.AffineStoreOp(
            iter_sum.result, rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        store_reg.attributes["to"] = StringAttr.get("sum_rv")

        # set terminator
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # restore insertion point
        GlobalInsertionPoint.restore()
        return rv


def make_affine_for(lb, ub, step=1, name="", stage="", reduction=False, ip=None):
    # Construct lower bound
    if isinstance(lb, int):
        lbCst = AffineConstantExpr.get(lb)
        lbMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[lbCst])
        lb_expr = None
    else:
        d0 = AffineDimExpr.get(0)
        lbMap = AffineMap.get(dim_count=1, symbol_count=0, exprs=[d0])
        lb_expr = lb.op
    lbMapAttr = AffineMapAttr.get(lbMap)

    # Construct upper bound
    if isinstance(ub, int):
        ubCst = AffineConstantExpr.get(ub)
        ubMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[ubCst])
        ub_expr = None
    else:
        d0 = AffineDimExpr.get(0)
        ubMap = AffineMap.get(dim_count=1, symbol_count=0, exprs=[d0])
        ub_expr = ub.op
    ubMapAttr = AffineMapAttr.get(ubMap)

    # Construct step
    if not isinstance(step, int):
        raise RuntimeError("Not supported")
    step = IntegerAttr.get(i32, step)

    # Create AffineForOp
    forOp = affine.AffineForOp(
        lb_expr,
        ub_expr,
        step,
        lbMapAttr,
        ubMapAttr,
        name=StringAttr.get(name),
        stage=("" if stage == "" else StringAttr.get(stage)),
        reduction=(IntegerAttr.get(i32, 1) if reduction else None),
        ip=ip,
    )

    return forOp


def make_if(cond, ip=None):
    # suppose in a imperative context (build in-place)
    d0 = AffineDimExpr.get(0)
    c1 = AffineConstantExpr.get(1)
    if_cond_set = IntegerSet.get(1, 0, [d0 - c1], [True])  # d0 - 1 == 0
    attr = IntegerSetAttr.get(if_cond_set)
    cast = CastOp(cond, idx_type)
    set_operands = [cast.result]

    if_op = affine.AffineIfOp(attr, set_operands, ip=ip)
    return if_op
