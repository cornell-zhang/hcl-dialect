#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from mlir.ir import *
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Any, Sequence


class AffineForOp:
    """Specialization for the Affine for op class."""

    def __init__(
        self,
        lower_bound,
        upper_bound,
        step,
        lowerBoundMap,
        upperBoundMap,
        reduction=None,
        iter_args: Sequence[Any] = [],
        name="",
        stage="",
        *,
        loc=None,
        ip=None
    ):
        """Creates an Affine `for` operation.
        operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                        (`step` integer-literal)? `{` op* `}`

        lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
        upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
        shorthand-bound ::= ssa-id | `-`? integer-literal

        - `lower_bound` is the value to use as lower bound of the loop.
        - `upper_bound` is the value to use as upper bound of the loop.
        - `step` is the value to use as loop step.
        - `iter_args` is a list of additional loop-carried arguments.
        """
        results = [arg.type for arg in iter_args]
        attributes = {}
        attributes["step"] = step
        attributes["lower_bound"] = lowerBoundMap
        attributes["upper_bound"] = upperBoundMap
        attributes["loop_name"] = name
        if stage != "":
            attributes["stage_name"] = stage
        if reduction:
            attributes["reduction"] = reduction
        if lower_bound == None and upper_bound == None:
            operands = list(iter_args)
        else:
            operands = [lower_bound, upper_bound] + list(iter_args)
        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=operands,
                attributes=attributes,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(IndexType.get(), *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variable(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Returns the loop-carried arguments usable within the loop.
        To obtain the loop-carried operands, use `iter_args`.
        """
        return self.body.arguments[1:]


class AffineStoreOp:
    def __init__(self, value, memref, indices, *, loc=None, ip=None):
        operands = []
        results = []
        operands.append(value)
        operands.append(memref)
        operands.extend(indices)
        attributes = {}
        identity_map = AffineMap.get_identity(len(indices))
        attr = AffineMapAttr.get(identity_map)
        attributes["map"] = attr
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                loc=loc,
                ip=ip,
            )
        )

    @property
    def value(self):
        return self.operation.operands[0]

    @property
    def memref(self):
        return self.operation.operands[1]

    @property
    def indices(self):
        _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
        return self.operation.operands[2 : 2 + _ods_variadic_group_length]
