#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .core import Value, Type
from .support import _FromCirctValue, _FromCirctType


class MutType(Type):
  """Type of a mutable elaboration-time cell holding a value of element_type."""

  def __init__(self, element_type: Type):
    self.element_type = element_type

  def __eq__(self, other) -> bool:
    return isinstance(other, MutType) and self.element_type == other.element_type

  def _codegen(self) -> ir.Type:
    return rtg.MutType.get(self.element_type._codegen())


class Mut(Value):
  """A mutable elaboration-time cell.

  Example::

    counter = Mut(Integer(0))
    n = counter.read()
    counter.write(n + Integer(1))
  """

  def __init__(self, initial: Value):
    self._value = rtg.MutCreateOp(initial._get_ssa_value())._get_ssa_value()

  def read(self) -> Value:
    """Read the current value of this mutable cell."""
    result = rtg.MutReadOp(self._value)._get_ssa_value()
    return _FromCirctValue(result)

  def write(self, new_value: Value) -> None:
    """Write a new value into this mutable cell."""
    rtg.MutWriteOp(self._value, new_value._get_ssa_value())

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return _FromCirctType(self._value.type)
