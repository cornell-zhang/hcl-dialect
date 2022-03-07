# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #


from hcl_mlir.dialects import affine, arith, builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import math, memref, std, scf
from hcl_mlir.ir import *


class HCLFlags(object):
    def __init__(self):
        self.BUILD_INPLACE = False
        self.EXTRACT_FUNCTION = False
        self.BIT_OP = False

    def enable_build_inplace(self):
        self.BUILD_INPLACE = True

    def disable_build_inplace(self):
        self.BUILD_INPLACE = False

    def is_build_inplace(self):
        return self.BUILD_INPLACE

    def reset(self):
        self.BUILD_INPLACE = False


flags = HCLFlags()
enable_build_inplace = flags.enable_build_inplace
disable_build_inplace = flags.disable_build_inplace
is_build_inplace = flags.is_build_inplace
reset_build_inplace = flags.reset
EXTRACT_FUNCTION = flags.EXTRACT_FUNCTION


def is_floating_point_type(dtype):
    return isinstance(dtype, (F16Type, F32Type, F64Type))


def is_integer_type(dtype):
    return isinstance(dtype, IntegerType)


def is_unsigned_type(dtype):
    return isinstance(dtype, IntegerType) and dtype.is_unsigned


def is_signed_type(dtype):
    return isinstance(dtype, IntegerType) and (dtype.is_unsigned or dtype.is_signless)


def is_fixed_type(dtype):
    return isinstance(dtype, (hcl_d.FixedType, hcl_d.UFixedType))


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
        if dtype[0:5] == "index":
            return IndexType.get()
        elif dtype[0:3] == "int":
            return IntegerType.get_signless(int(dtype[3:]))
        elif dtype[0:4] == "uint":
            return IntegerType.get_unsigned(int(dtype[4:]))
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
            return hcl_d.FixedType.get(int(strs[0]), int(strs[1]))
        elif dtype[0:6] == "ufixed":
            strs = dtype[6:].split("_")
            return hcl_d.UFixedType.get(int(strs[0]), int(strs[1]))
        else:
            raise RuntimeError("Unrecognized data type")
    else:
        raise RuntimeError("Unrecognized data type format")


def get_concrete_type(dtype):
    if IntegerType.isinstance(dtype):
        return IntegerType(dtype)
    elif F16Type.isinstance(dtype):
        return F16Type(dtype)
    elif F32Type.isinstance(dtype):
        return F32Type(dtype)
    elif F64Type.isinstance(dtype):
        return F64Type(dtype)
    elif hcl_d.FixedType.isinstance(dtype):
        return hcl_d.FixedType(dtype)
    elif hcl_d.UFixedType.isinstance(dtype):
        return hcl_d.UFixedType(dtype)
    else:
        raise RuntimeError("Unrecognized data type")


def get_bitwidth(dtype):
    if IntegerType.isinstance(dtype):
        return dtype.width
    elif F16Type.isinstance(dtype):
        return 16
    elif F32Type.isinstance(dtype):
        return 32
    elif F64Type.isinstance(dtype):
        return 64
    elif hcl_d.FixedType.isinstance(dtype):
        return dtype.width
    elif hcl_d.UFixedType.isinstance(dtype):
        return dtype.width
    else:
        raise RuntimeError("Unrecognized data type")


def print_mlir_type(dtype):
    if is_floating_point_type(dtype):
        if dtype.width == 32:
            return "float"
        elif dtype.width == 64:
            return "double"
        else:
            raise RuntimeError("Not supported data type")
    elif is_integer_type(dtype):
        if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
            if dtype.width == 32:
                return "int"
            elif dtype.width == 64:
                return "long int"
            elif dtype.width == 1:
                return "bool"
            else:
                return "ap_int<{}>".format(dtype.width)
        elif dtype.is_unsigned:
            if dtype.width == 32:
                return "unsigned int"
            elif dtype.width == 64:
                return "unsigned long int"
            else:
                return "ap_uint<{}>".format(dtype.width)
    elif is_fixed_type(dtype):
        if isinstance(dtype, hcl_d.FixedType):
            return "ap_fixed<{}, {}>".format(dtype.width, dtype.frac)
        elif isinstance(dtype, hcl_d.UFixedType):
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


def get_hcl_op(expr, dtype=None):
    if isinstance(expr, (int, float)):
        if dtype == None:
            if isinstance(expr, int):
                return ConstantOp(IntegerType.get_signless(32), expr)
            elif isinstance(expr, float):
                return ConstantOp(F32Type.get(), expr)
        else:
            return ConstantOp(dtype, expr)
    else:
        if dtype != None and dtype != expr.dtype:
            expr = CastOp(expr, dtype)
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
        res_type = F64Type.get()
    # 3) Otherwise, if one operand is float
    elif isinstance(ltype, F32Type):
        # integer type to float (the only real type possible is float, which remains as-is)
        res_type = F32Type.get()
    # 4) Otherwise, both operands are integers. Both operands undergo integer promotions (see below); then, after integer promotion, one of the following cases applies:
    elif isinstance(ltype, (IntegerType, IndexType)):
        res_type = ltype
    # 5) Fixed types
    elif is_fixed_type(ltype):
        # TODO: UFixed type
        if is_integer_type(rtype):
            res_type = hcl_d.FixedType.get(rtype.width, ltype.frac)
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
        res_type = hcl_d.FixedType.get(rwidth, lhs.dtype.frac)
        cast = CastOp(lhs, res_type)
        return cast, rhs
    elif lwidth > rwidth:
        res_type = hcl_d.FixedType.get(lwidth, rhs.dtype.frac)
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
    def result(self):  # get_op_result_or_value
        if isinstance(self.built_op, BlockArgument):
            return self.built_op
        else:
            return self.built_op.result

    @staticmethod
    def generic_op(OpClass, lhs, rhs, arg=None):
        # turn py builtin op to hcl op
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)

        # type checking & conversion
        # get_type_rank has the valid type checking
        lhs.dtype = get_mlir_type(lhs.dtype)
        rhs.dtype = get_mlir_type(rhs.dtype)
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

        # type checking & conversion
        if lhs.dtype != rhs.dtype:
            rhs = CastOp(rhs, lhs.dtype)
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
        return self.generic_op(CmpOp, self, other, arg="lt")

    def __le__(self, other):
        return self.generic_op(CmpOp, self, other, arg="le")

    def __eq__(self, other):
        return self.generic_op(CmpOp, self, other, arg="eq")

    def __ne__(self, other):
        return self.generic_op(CmpOp, self, other, arg="ne")

    def __gt__(self, other):
        return self.generic_op(CmpOp, self, other, arg="gt")

    def __ge__(self, other):
        return self.generic_op(CmpOp, self, other, arg="ge")

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

    def __init__(self, op, name=""):
        super().__init__(op, dtype="index")
        self.name = name
        self.built_op = op

    def update_op(self, op):
        self.op = op
        self.built_op = op


class ReduceVar(IterVar):
    """reduce_axis
    induction variable of reduction loop
    """

    def __init__(self, op, bound=None, name=""):
        super().__init__(op, name)
        self.bound = bound

    @property
    def lower_bound(self):
        return self.bound[0]

    @property
    def upper_bound(self):
        return self.bound[1]


class ConstantOp(ExprOp):
    def __init__(self, dtype, val):
        super().__init__(arith.ConstantOp)
        self.val = val
        self.dtype = get_mlir_type(dtype)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if not is_fixed_type(self.dtype):
            if isinstance(self.dtype, IntegerType):
                if self.dtype.width == 1:
                    value_attr = BoolAttr.get(self.val)
                else:
                    value_attr = IntegerAttr.get(
                        IntegerType.get_signless(self.dtype.width), self.val
                    )
            elif isinstance(self.dtype, F32Type):
                value_attr = FloatAttr.get(F32Type.get(), self.val)
            elif isinstance(self.dtype, IndexType):
                value_attr = IntegerAttr.get(IndexType.get(), self.val)
            else:
                raise RuntimeError("Type error")
            if is_unsigned_type(self.dtype):
                dtype = IntegerType.get_signless(self.dtype.width)
                self.built_op = self.op(
                    dtype, value_attr, ip=GlobalInsertionPoint.get()
                )
                self.built_op.attributes["unsigned"] = UnitAttr.get()
            else:
                self.built_op = self.op(
                    self.dtype, value_attr, ip=GlobalInsertionPoint.get()
                )
            return self.built_op
        else:  # fixed types
            if isinstance(self.val, float):
                value_attr = FloatAttr.get(F32Type.get(), self.val)
                self.built_op = self.op(
                    F32Type.get(), value_attr, ip=GlobalInsertionPoint.get()
                )
            elif isinstance(self.val, int):
                value_attr = IntegerAttr.get(IntegerType.get_signless(32), self.val)
                self.built_op = self.op(
                    IntegerType.get_signless(32),
                    value_attr,
                    ip=GlobalInsertionPoint.get(),
                )
            else:
                raise RuntimeError("Type error")
            return self.built_op


class TensorSlice(ExprOp):
    def __init__(self, full_shape, op, dtype, parent, indices, name=None):
        super().__init__(op)
        self.op = op
        self.full_shape = full_shape
        self.dtype = dtype
        self.name = name
        self.parent = parent
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
                self.full_shape,
                self.op,
                self.dtype,
                self.parent,
                self.indices + indices,
                self.name,
            )
        elif len(self.indices + indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            load = LoadOp(self.parent, new_indices)
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
            # TODO(Niansong): I think this is doable actually
            raise RuntimeError("Writing to a slice of tensor is not allowed.")
        elif len(self.indices + indices) == len(self.shape):
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            return StoreOp(expr, self.parent, self.indices + new_indices)
        else:
            raise RuntimeError("Indices length > # of array dimensions")


class TensorOp(ExprOp):
    def __init__(self, shape, op, dtype, name=None):
        if op != memref.AllocOp and not isinstance(op, BlockArgument):
            raise RuntimeError("Not supported TensorOp. Got {}".format(op))
        super().__init__(op)
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def build(self):
        if self.op == memref.AllocOp:
            self.built_op = self.op(
                self.memref_type, [], [], None, ip=GlobalInsertionPoint.get()
            )
            if is_unsigned_type(self.dtype):
                self.built_op.attributes["unsigned"] = UnitAttr.get()
            self.built_op.attributes["name"] = StringAttr.get(self.name)
        elif isinstance(self.op, BlockArgument):
            self.built_op = self.op
        else:
            raise RuntimeError(
                "TensorOp should use memref.AllocOp or BlockArgument to implement. Got {}".format(
                    self.op
                )
            )
        return self.built_op

    def update_op(self, op):
        self.built_op = op

    @property
    def memref_type(self):
        dtype = get_mlir_type(self.dtype)
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
        else:
            dtype = self.dtype
        return MemRefType.get(self.shape, dtype)

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
            return TensorSlice(
                self.shape, self.op, self.dtype, self, indices, self.name
            )
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            load = LoadOp(self, new_indices)
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
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            expr = get_hcl_op(expr)
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
        self.built_op = self.op(self.val.result, ip=GlobalInsertionPoint.get())
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
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
            self.lhs.result, self.rhs.result, ip=GlobalInsertionPoint.get()
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class CmpOp(BinaryOp):
    """
    # Check mlir/Dialect/Arithmetic/IR/ArithmeticBase.td
    # s/u: signed/unsigned
    # o/u: ordered/unordered
    #      ordered means only one of < = > cases is true
    #      unordered happens for floating points with NaN
    // Opcode              U L G E    Intuitive operation
    FCMP_FALSE =  0,  ///< 0 0 0 0    Always false (always folded)
    FCMP_OEQ   =  1,  ///< 0 0 0 1    True if ordered and equal
    FCMP_OGT   =  2,  ///< 0 0 1 0    True if ordered and greater than
    FCMP_OGE   =  3,  ///< 0 0 1 1    True if ordered and greater than or equal
    FCMP_OLT   =  4,  ///< 0 1 0 0    True if ordered and less than
    FCMP_OLE   =  5,  ///< 0 1 0 1    True if ordered and less than or equal
    FCMP_ONE   =  6,  ///< 0 1 1 0    True if ordered and operands are unequal
    FCMP_ORD   =  7,  ///< 0 1 1 1    True if ordered (no nans)
    FCMP_UNO   =  8,  ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    FCMP_UEQ   =  9,  ///< 1 0 0 1    True if unordered or equal
    FCMP_UGT   = 10,  ///< 1 0 1 0    True if unordered or greater than
    FCMP_UGE   = 11,  ///< 1 0 1 1    True if unordered, greater than, or equal
    FCMP_ULT   = 12,  ///< 1 1 0 0    True if unordered or less than
    FCMP_ULE   = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
    FCMP_UNE   = 14,  ///< 1 1 1 0    True if unordered or not equal
    FCMP_TRUE  = 15,  ///< 1 1 1 1    Always true (always folded)
    """

    ATTR_MAP = {
        "int": {
            "eq": 0,
            "ne": 1,
            "slt": 2,
            "sle": 3,
            "sgt": 4,
            "sge": 5,
            "ult": 6,
            "ule": 7,
            "ugt": 8,
            "uge": 9,
        },
        "float": {
            "false": 0,
            "oeq": 1,
            "ogt": 2,
            "oge": 3,
            "olt": 4,
            "ole": 5,
            "one": 6,
            "ord": 7,
            "ueq": 8,
            "ugt": 9,
            "uge": 10,
            "ult": 11,
            "ule": 12,
            "une": 13,
            "uno": 14,
            "true": 15,
        },
        "fixed": {
            "eq": 0,
            "ne": 1,
            "slt": 2,
            "sle": 3,
            "sgt": 4,
            "sge": 5,
            "ult": 6,
            "ule": 7,
            "ugt": 8,
            "uge": 9,
        },
    }

    def __init__(self, lhs, rhs, arg):
        self.arg = arg
        dtype = lhs.dtype
        if is_integer_type(dtype) or is_index_type(dtype):
            self.op = arith.CmpIOp
            if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
                self.arg = CmpOp.ATTR_MAP["int"][
                    "s" + arg if arg not in ["eq", "ne"] else arg
                ]
            else:
                self.arg = CmpOp.ATTR_MAP["int"][
                    "u" + arg if arg not in ["eq", "ne"] else arg
                ]
        elif is_floating_point_type(dtype):
            self.op = arith.CmpFOp
            self.arg = CmpOp.ATTR_MAP["float"]["o" + arg]
        elif is_fixed_type(dtype):
            self.op = hcl_d.CmpFixedOp
            if isinstance(dtype, hcl_d.FixedType):
                self.arg = CmpOp.ATTR_MAP["fixed"][
                    "s" + arg if arg not in ["eq", "ne"] else arg
                ]
            else:
                self.arg = CmpOp.ATTR_MAP["fixed"][
                    "u" + arg if arg not in ["eq", "ne"] else arg
                ]
        else:
            raise RuntimeError("Unsupported types")
        super().__init__(self.op, IntegerType.get_signless(1), lhs, rhs)

    def build(self):
        self.built_op = self.op(
            self.dtype,
            IntegerAttr.get(IntegerType.get_signless(64), self.arg),
            self.lhs.result,
            self.rhs.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class AddOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": arith.AddFOp, "int": arith.AddIOp, "fixed": hcl_d.AddFixedOp},
            dtype,
            lhs,
            rhs,
        )


class SubOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": arith.SubFOp, "int": arith.SubIOp, "fixed": hcl_d.SubFixedOp},
            dtype,
            lhs,
            rhs,
        )


class MulOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": arith.MulFOp, "int": arith.MulIOp, "fixed": hcl_d.MulFixedOp},
            dtype,
            lhs,
            rhs,
        )


class DivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"float": arith.DivFOp, "int": arith.DivUIOp}, dtype, lhs, rhs)


class FloorDivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__(
            {"float": arith.DivFOp, "int": arith.DivUIOp},  # not supported!
            dtype,
            lhs,
            rhs,
        )


class RemOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs):
        super().__init__({"float": arith.RemFOp, "int": arith.RemUIOp}, dtype, lhs, rhs)


class LeftShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(arith.ShLIOp, lhs.dtype, lhs, rhs)


class RightShiftOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(arith.ShRUIOp, lhs.dtype, lhs, rhs)


class AndOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(arith.AndIOp, lhs.dtype, lhs, rhs)


class OrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(arith.OrIOp, lhs.dtype, lhs, rhs)


class XOrOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(arith.XOrIOp, lhs.dtype, lhs, rhs)


class NegOp(UnaryOp):
    def __init__(self, dtype, val):
        super().__init__(
            {"float": arith.NegFOp, "int": arith.NegFOp}, dtype, val
        )  # use the same op


class BitCastOp(UnaryOp):
    def __init__(self, dtype, val):
        super().__init__(arith.BitcastOp, dtype, val)

    def build(self):
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
            self.built_op = self.op(
                dtype, self.val.result, ip=GlobalInsertionPoint.get()
            )
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        else:
            dtype = self.dtype
            self.built_op = self.op(
                self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class MathExpOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.ExpOp, F32Type.get(), val)


class MathPowOp(BinaryOp):
    def __init__(self, x, y):
        if not isinstance(x, (int, float)):
            dtype = x.dtype
        if not isinstance(y, (int, float)):
            dtype = y.dtype
        x = get_hcl_op(x, F32Type.get())
        y = get_hcl_op(y, F32Type.get())
        super().__init__(math.PowFOp, dtype, x, y)

    def build(self):
        self.built_op = self.op(
            self.lhs.result, self.rhs.result, ip=GlobalInsertionPoint.get()
        )
        if not is_floating_point_type(self.dtype):
            self.built_op = arith.FPToSIOp(
                self.dtype, self.built_op.result, ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class MathLogOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.LogOp, F32Type.get(), val)


class MathLog2Op(UnaryOp):
    def __init__(self, val):
        super().__init__(math.Log2Op, F32Type.get(), val)


class MathLog10Op(UnaryOp):
    def __init__(self, val):
        super().__init__(math.Log10Op, F32Type.get(), val)


class MathSqrtOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.SqrtOp, F32Type.get(), val)


class MathSinOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.SinOp, F32Type.get(), val)


class MathCosOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.CosOp, F32Type.get(), val)


class MathTanhOp(UnaryOp):
    def __init__(self, val):
        super().__init__(math.TanhOp, F32Type.get(), val)


class CastOp(ExprOp):
    def __init__(self, val, res_type=None):
        # dtype is the result type
        res_type = get_mlir_type(res_type)
        self.val = get_hcl_op(val)
        self.val.dtype = get_mlir_type(self.val.dtype)
        if res_type == self.val.dtype:
            op = None
        elif (is_index_type(res_type) and is_integer_type(self.val.dtype)) or (
            is_index_type(self.val.dtype) and is_integer_type(res_type)
        ):
            op = arith.IndexCastOp
        elif is_signed_type(self.val.dtype) and is_floating_point_type(res_type):
            op = arith.SIToFPOp
        elif is_unsigned_type(self.val.dtype) and is_floating_point_type(res_type):
            op = arith.UIToFPOp
        elif is_signed_type(res_type) and is_floating_point_type(self.val.dtype):
            op = arith.FPToSIOp
        elif is_unsigned_type(res_type) and is_floating_point_type(self.val.dtype):
            op = arith.FPToUIOp
        elif is_integer_type(res_type) and is_integer_type(self.val.dtype):
            if res_type.width < self.val.dtype.width:
                op = arith.TruncIOp
            elif res_type.width == self.val.dtype.width:
                op = None
            else:
                if not is_unsigned_type(res_type):
                    op = arith.ExtSIOp
                else:
                    op = arith.ExtUIOp
        elif is_floating_point_type(res_type) and is_floating_point_type(
            self.val.dtype
        ):
            if res_type.width < self.val.dtype.width:
                op = arith.TruncFOp
            elif res_type.width == self.val.dtype.width:
                op = None
            else:
                op = arith.ExtFOp
        else:
            op = builtin.UnrealizedConversionCastOp
        super().__init__(op, res_type)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if self.op in [
            arith.IndexCastOp,
            arith.SIToFPOp,
            arith.UIToFPOp,
            arith.FPToSIOp,
            arith.FPToUIOp,
            arith.TruncIOp,
            arith.TruncFOp,
            arith.ExtUIOp,
            arith.ExtSIOp,
            arith.ExtFOp,
        ]:
            if is_unsigned_type(self.dtype):
                dtype = IntegerType.get_signless(self.dtype.width)
                self.built_op = self.op(
                    dtype, self.val.result, ip=GlobalInsertionPoint.get()
                )
                self.built_op.attributes["unsigned"] = UnitAttr.get()
            else:
                self.built_op = self.op(
                    self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
                )
        elif self.op == None:
            self.built_op = self.val.built_op
        else:  # builtin.UnrealizedConversionCastOp
            self.built_op = self.op(
                [self.dtype], [self.val.result], ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class GetBitOp(ExprOp):
    def __init__(self, num, index):
        super().__init__(hcl_d.GetIntBitOp, IntegerType.get_signless(1))
        self.num = num
        if isinstance(index, int):
            index = ConstantOp(IndexType.get(), index)
        self.index = index
        if not isinstance(self.index.dtype, IndexType):
            print(
                "Warning: GetBitOp's input is not an index. Cast from {} to {}.".format(
                    self.index.dtype, IndexType.get()
                )
            )
            self.index = CastOp(self.index, IndexType.get())
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype,
            self.num.result,
            self.index.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        flags.BIT_OP = True
        return self.built_op


class SetBitOp(ExprOp):
    def __init__(self, num, index, val):
        super().__init__(hcl_d.SetIntBitOp, None)  # No return value!
        self.num = num  # actually a LoadOp
        if isinstance(index, int):
            index = ConstantOp(IndexType.get(), index)
        self.index = index
        if val not in [0, 1]:
            raise RuntimeError("Can only set a bit of 0/1. Got {}.".format(val))
        if isinstance(val, int):
            val = ConstantOp(IntegerType.get_signless(1), val)
        self.val = val
        if not isinstance(self.index.dtype, IndexType):
            print(
                "Warning: SetBitOp's input is not an index. Cast from {} to {}.".format(
                    self.index.dtype, IndexType.get()
                )
            )
            self.index = CastOp(self.index, IndexType.get())
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.num.result,
            self.index.result,
            self.val.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        self.built_op = StoreOp(self.num, self.num.tensor, self.num.indices)
        flags.BIT_OP = True
        return self.built_op


class LoadOp(ExprOp):
    def __init__(self, tensor, indices):
        super().__init__(affine.AffineLoadOp, tensor.dtype)
        self.tensor = tensor
        self.indices = []
        for index in indices:
            if not isinstance(index.dtype, IndexType):
                print(
                    "Warning: LoadOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                )
                index = CastOp(index, IndexType.get())
            self.indices.append(index)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        # test if affine expressions
        visitor = ASTVisitor(mode="profile")
        exprs = []
        flag = True
        for index in self.indices:
            try:
                affine_expr = visitor.visit_affine_expr(index)
                exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            affine_map = AffineMap.get(
                dim_count=len(visitor.iv), symbol_count=0, exprs=exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            self.built_op = self.op(
                self.tensor.result,
                visitor.iv,
                affine_attr,
                ip=GlobalInsertionPoint.get(),
            )
        else:
            new_indices = []
            for index in self.indices:
                new_indices.append(index.result)
            self.built_op = memref.LoadOp(
                self.tensor.result, new_indices, ip=GlobalInsertionPoint.get()
            )
        self.built_op.attributes["from"] = StringAttr.get(self.tensor.name)
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
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

    def __setitem__(self, bit_idx, expr):
        if not is_integer_type(self.dtype):
            raise RuntimeError("Only fixed integers can access the bits")
        if not isinstance(bit_idx, tuple):
            bit_idx = (bit_idx,)
        if not len(bit_idx) == 1:
            raise RuntimeError("Can only access one bit of the integer")
        bit_idx = bit_idx[0]
        return SetBitOp(self, bit_idx, expr)


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
        self.indices = []
        for index in indices:
            if not isinstance(index.dtype, IndexType):
                print(
                    "Warning: LoadOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                )
                index = CastOp(index, IndexType.get())
            self.indices.append(index)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        # test if affine expressions
        visitor = ASTVisitor(mode="profile")
        exprs = []
        flag = True
        for index in self.indices:
            try:
                affine_expr = visitor.visit_affine_expr(index)
                exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            affine_map = AffineMap.get(
                dim_count=len(visitor.iv), symbol_count=0, exprs=exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            self.built_op = self.op(
                self.val.result,
                self.to_tensor.result,
                visitor.iv,
                affine_attr,
                ip=GlobalInsertionPoint.get(),
            )
        else:
            new_indices = []
            for index in self.indices:
                new_indices.append(index.result)
            self.built_op = memref.StoreOp(
                self.val.result,
                self.to_tensor.result,
                new_indices,
                ip=GlobalInsertionPoint.get(),
            )
        self.built_op.attributes["to"] = StringAttr.get(self.to_tensor.name)
        if is_unsigned_type(self.to_tensor.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
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
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
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
            self.cond.result,
            self.true_val.result,
            self.false_val.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class ReduceOp(ExprOp):
    # cannot build inplace!!!
    def __init__(self, op, axis, dtype):
        super().__init__(op, dtype=get_mlir_type(dtype))
        self.axis = axis


class SumOp(ReduceOp):
    def __init__(self, op, axis, dtype):
        super().__init__(op, axis, dtype)


class MinOp(ReduceOp):
    def __init__(self, op, axis, dtype):
        super().__init__(op, axis, dtype)


class MaxOp(ReduceOp):
    def __init__(self, op, axis, dtype):
        super().__init__(op, axis, dtype)


class ASTVisitor:
    def __init__(self, mode="build"):
        self.iv = []
        if mode not in ["build", "remove", "profile"]:
            raise RuntimeError(
                "ASTVisitor only supports build, remove, or profile mode"
            )
        self.mode = mode
        self.load = []
        self.store = []

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
        elif isinstance(expr, SetBitOp):
            return self.visit_setbit_op(expr)
        elif isinstance(expr, CastOp):
            return self.visit_cast_op(expr)
        elif isinstance(expr, ReduceOp):
            return self.visit_reduce_op(expr)
        elif isinstance(expr, ConstantOp):
            return self.visit_constant_op(expr)
        else:  # IterVar
            return expr.built_op  # BlockArgument

    def visit_affine_expr(self, expr):
        """Build affine expression.
        * Should all be binary op
        * AffineExpr can be automatically simplied
        """
        if not isinstance(expr, (IterVar, ConstantOp, CastOp, BinaryOp)):
            raise RuntimeError("Not an affine index!")
        if isinstance(expr, IterVar):
            if expr.op not in self.iv:
                self.iv.append(expr.op)  # BlockArgument
                return AffineExpr.get_dim(len(self.iv) - 1)
            else:
                return AffineExpr.get_dim(self.iv.index(expr.op))
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
            return AffineExpr.get_floor_div(lhs, rhs)  # or get_ceil_div
        elif isinstance(expr, RemOp):
            return lhs % rhs
        else:
            raise RuntimeError("Not an affine index!")

    def visit_unary_op(self, expr):
        if self.mode == "build":
            self.visit(expr.val)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.val)
        else:
            expr.built_op.operation.erase()
            self.visit(expr.val)

    def visit_binary_op(self, expr):
        if self.mode == "build":
            self.visit(expr.lhs)
            self.visit(expr.rhs)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.lhs)
            self.visit(expr.rhs)
        else:
            expr.built_op.operation.erase()
            self.visit(expr.rhs)
            self.visit(expr.lhs)

    def visit_ternary_op(self, expr):
        if self.mode == "build":
            self.visit(expr.cond)
            self.visit(expr.true_val)
            self.visit(expr.false_val)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.cond)
            self.visit(expr.true_val)
            self.visit(expr.false_val)
        else:
            expr.built_op.operation.erase()
            self.visit(expr.false_val)
            self.visit(expr.true_val)
            self.visit(expr.cond)

    def visit_load_op(self, expr):
        if self.mode == "remove":
            expr.built_op.operation.erase()
            return
        elif self.mode == "profile":
            self.load.append(expr)
            return
        else:
            return expr.build()

    def visit_store_op(self, expr):
        if self.mode == "remove":
            expr.built_op.operation.erase()
            return
        elif self.mode == "profile":
            self.store.append(expr)
            return
        else:
            return expr.build()

    def visit_cast_op(self, expr):
        if self.mode == "remove":
            expr.built_op.operation.erase()
        self.visit(expr.val)
        if self.mode == "build":
            return expr.build()

    def visit_getbit_op(self, expr):
        if self.mode == "build":
            self.visit(expr.num)
            self.visit(expr.index)
            return expr.build()
        elif self.mode == "remove":
            expr.built_op.operation.erase()
            self.visit(expr.index)
            self.visit(expr.num)
        else:
            self.visit(expr.num)
            self.visit(expr.index)

    def visit_setbit_op(self, expr):
        if self.mode == "remove":
            expr.built_op.operation.erase()
        self.visit(expr.num)
        self.visit(expr.index)
        self.visit(expr.val)
        if self.mode == "build":
            return expr.build()

    def visit_constant_op(self, expr):
        if self.mode == "build":
            return expr.build()
        elif self.mode == "profile":
            pass
        else:
            expr.built_op.operation.erase()

    def visit_reduce_op(self, expr):
        if self.mode == "remove":
            raise RuntimeError("Cannot remove ReduceOp")
        elif self.mode == "profile":
            return
        # save insetion point
        save_ip = GlobalInsertionPoint.get()

        # create a single-element register for reduction
        dtype = expr.dtype
        if is_unsigned_type(dtype):
            dtype = IntegerType.get_signless(dtype.width)
        memref_type = MemRefType.get((1,), dtype)
        rv = memref.AllocOp(memref_type, [], [], None, ip=GlobalInsertionPoint.get())
        if isinstance(expr, SumOp):
            prefix = "sum"
            init_val = 0
            reduce_op = {
                "float": arith.AddFOp,
                "si": arith.AddIOp,
                "ui": arith.AddIOp,
                "fixed": hcl_d.AddFixedOp,
            }
        elif isinstance(expr, MinOp):
            prefix = "min"
            init_val = 0x3F3F3F3F  # magic large number
            reduce_op = {
                "float": arith.MinFOp,
                "si": arith.MinSIOp,
                "ui": arith.MinUIOp,
                "fixed": hcl_d.MinFixedOp,
            }
        elif isinstance(expr, MaxOp):
            prefix = "max"
            init_val = -0x3F3F3F3F
            reduce_op = {
                "float": arith.MaxFOp,
                "si": arith.MaxSIOp,
                "ui": arith.MaxUIOp,
                "fixed": hcl_d.MaxFixedOp,
            }
        else:
            raise RuntimeError("Should not in this branch")
        rv.attributes["name"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            rv.attributes["unsigned"] = UnitAttr.get()
        # initialize the single-element register
        zero_idx = arith.ConstantOp(
            IndexType.get(),
            IntegerAttr.get(IndexType.get(), 0),
            ip=GlobalInsertionPoint.get(),
        )
        # initialize the original value of the reducer
        if is_floating_point_type(dtype):
            zero_value = arith.ConstantOp(
                dtype, FloatAttr.get(dtype, init_val), ip=GlobalInsertionPoint.get()
            )
        elif is_integer_type(dtype):
            zero_value = arith.ConstantOp(
                dtype, IntegerAttr.get(dtype, init_val), ip=GlobalInsertionPoint.get()
            )
        elif is_fixed_type(dtype):
            value_attr = IntegerAttr.get(IntegerType.get_signless(32), init_val)
            zero_value = arith.ConstantOp(
                IntegerType.get_signless(32), value_attr, ip=GlobalInsertionPoint.get()
            )
            zero_value = builtin.UnrealizedConversionCastOp(
                [dtype], [zero_value.result], ip=GlobalInsertionPoint.get()
            )
        else:
            raise RuntimeError("Unrecognized data type in reduction op")
        if is_unsigned_type(expr.dtype):
            zero_value.attributes["unsigned"] = UnitAttr.get()

        store = affine.AffineStoreOp(
            zero_value.result,
            rv.result,
            [zero_idx.result],
            ip=GlobalInsertionPoint.get(),
        )
        store.attributes["to"] = StringAttr.get("{}_rv".format(prefix))

        # create reduction loop
        if not isinstance(expr.axis, list):
            new_axes = [expr.axis]
        else:
            new_axes = expr.axis
        body_ip = GlobalInsertionPoint.get()
        for axis in new_axes:
            reduction_loop = make_affine_for(
                axis.lower_bound,
                axis.upper_bound,
                step=1,
                name=axis.name,
                ip=body_ip,
            )
            body_ip = InsertionPoint(reduction_loop.body)

            # update reduction variable
            axis.update_op(reduction_loop.induction_variable)

            # update insertion point
            GlobalInsertionPoint.save(body_ip)

        # visit subexpressions
        data = self.visit(expr.op)

        # load register value and reduce
        load = affine.AffineLoadOp(
            rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        load.attributes["from"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            load.attributes["unsigned"] = UnitAttr.get()
        if is_floating_point_type(dtype):
            reduce_op = reduce_op["float"]
        elif is_integer_type(dtype):
            if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
                reduce_op = reduce_op["si"]
            else:  # unsigned
                reduce_op = reduce_op["ui"]
        elif is_fixed_type(dtype):
            reduce_op = reduce_op["fixed"]
        else:
            raise RuntimeError("Unsupported type")
        if dtype != data.result.type:
            print(
                "Warning: Reduction variable should have the same type with the data. Got {} and {}. Do type casting...".format(
                    dtype, data.result.type
                )
            )
            placeholder = LoadOp(expr, [])
            placeholder.built_op = data
            data = CastOp(placeholder, dtype)
            data.build()
        iter_reduction = reduce_op(
            data.result, load.result, ip=GlobalInsertionPoint.get()
        )
        if is_unsigned_type(expr.dtype):
            iter_reduction.attributes["unsigned"] = UnitAttr.get()

        # store the result back to register
        store_reg = affine.AffineStoreOp(
            iter_reduction.result,
            rv.result,
            [zero_idx.result],
            ip=GlobalInsertionPoint.get(),
        )
        store_reg.attributes["to"] = StringAttr.get("{}_rv".format(prefix))

        # set terminator
        for axis in new_axes:
            affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())
            # restore insertion point
            GlobalInsertionPoint.restore()

        zero_idx = arith.ConstantOp(
            IndexType.get(),
            IntegerAttr.get(IndexType.get(), 0),
            ip=GlobalInsertionPoint.get(),
        )
        value = affine.AffineLoadOp(
            rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        value.attributes["from"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            value.attributes["unsigned"] = UnitAttr.get()
        expr.built_op = value
        return value


def make_affine_for(
    lb, ub, step=1, name="", stage="", reduction=False, ip=None, loc=None
):
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
    step = IntegerAttr.get(IntegerType.get_signless(32), step)

    # Create AffineForOp
    forOp = affine.AffineForOp(
        lb_expr,
        ub_expr,
        step,
        lbMapAttr,
        ubMapAttr,
        name=(StringAttr.get("") if name in ["", None] else StringAttr.get(name)),
        stage=("" if stage == "" else StringAttr.get(stage)),
        reduction=(
            IntegerAttr.get(IntegerType.get_signless(32), 1) if reduction else None
        ),
        ip=ip,
        loc=loc,
    )

    return forOp


def make_if(cond, ip=None):
    # suppose in a imperative context (build in-place)
    if not isinstance(cond, CmpOp):
        raise RuntimeError("`if` operation condition should be CmpOp")
    visitor = ASTVisitor(mode="profile")
    visitor.visit(cond)
    if len(visitor.load) != 0 or len(visitor.store) != 0:
        remover = ASTVisitor(mode="remove")
        remover.visit(cond)
        builder = ASTVisitor(mode="build")
        builder.visit(cond)
        if_op = scf.IfOp([], cond.result, ip=ip)
    else:  # Affine expression
        if not isinstance(cond.lhs.dtype, (IntegerType, IndexType)) or not isinstance(
            cond.rhs.dtype, (IntegerType, IndexType)
        ):
            raise RuntimeError("`affine.if` can only support integer comparison")
        # only support affine expressions now (i.e. calculations on iteration variables)
        if cond.arg == 0:  # eq
            # lhs==rhs
            eq_flags = [True]
            new_conds = [cond.lhs - cond.rhs]
        elif cond.arg == 1:  # ne
            # lhs>rhs and lhs<rhs
            raise RuntimeError("Not supported for `affine.if`")
        elif cond.arg == 2:  # slt
            # lhs<rhs -> rhs-lhs>0 -> rhs-lhs>=1 -> rhs-lhs-1>=0
            eq_flags = [False]
            new_conds = [cond.rhs - cond.lhs - ConstantOp(cond.lhs.dtype, 1)]
        elif cond.arg == 3:  # sle
            # lhs<=rhs -> rhs-lhs>=0
            eq_flags = [False]
            new_conds = [cond.rhs - cond.lhs]
        elif cond.arg == 4:  # sgt
            # lhs>rhs -> lhs-rhs-1>=0
            eq_flags = [False]
            new_conds = [cond.lhs - cond.rhs - ConstantOp(cond.lhs.dtype, 1)]
        elif cond.arg == 5:  # sge
            # lhs>=rhs -> lhs-rhs>=0
            eq_flags = [False]
            new_conds = [cond.lhs - cond.rhs]
        else:
            raise RuntimeError("Predicate of CmpOp")

        cond.built_op.operation.erase()
        exprs = []
        # make sure all the AffineExpr are referenced in one visitor
        builder = ASTVisitor(mode="build")
        for new_cond in new_conds:
            remover = ASTVisitor(mode="remove")
            remover.visit(new_cond)
            # rebuild condition
            exprs.append(builder.visit_affine_expr(new_cond))
        if_cond_set = IntegerSet.get(len(builder.iv), 0, exprs, eq_flags)
        attr = hcl_d.IntegerSetAttr.get(if_cond_set)

        if_op = affine.AffineIfOp(attr, builder.iv, ip=ip)

    return if_op


def make_while(cond, ip=None):
    # suppose in a imperative context (build in-place)
    if not isinstance(cond, CmpOp):
        raise RuntimeError("`if` operation condition should be CmpOp")
    while_op = scf.WhileOp([], [], ip=ip)
    while_op.before.blocks.append(*[])
    while_op.after.blocks.append(*[])
    GlobalInsertionPoint.save(while_op.before.blocks[0])
    remover = ASTVisitor(mode="remove")
    remover.visit(cond)
    builder = ASTVisitor(mode="build")
    builder.visit(cond)
    scf.ConditionOp(cond.result, [], ip=GlobalInsertionPoint.get())
    GlobalInsertionPoint.restore()
    return while_op


def get_affine_loop_nests(func):
    results = []
    for op in func.entry_block.operations:
        if isinstance(op, affine.AffineForOp):  # outer-most
            band = []
            loop = op
            while True:
                band.append({"name": loop.attributes["loop_name"], "body": loop})
                for loop in loop.body.operations:
                    if isinstance(loop, affine.AffineForOp):
                        break
                else:
                    break
            results.append(band)
    return results
