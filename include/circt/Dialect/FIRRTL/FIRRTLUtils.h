//===- FIRRTLUtils.h - FIRRTL IR Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Parallel.h"

namespace circt {
namespace firrtl {
/// Emit a connect between two values.
void emitConnect(OpBuilder &builder, Location loc, Value lhs, Value rhs);
void emitConnect(ImplicitLocOpBuilder &builder, Value lhs, Value rhs);

/// Utility for generating a constant attribute.
IntegerAttr getIntAttr(Type type, const APInt &value);

/// Utility for generating a constant zero attribute.
IntegerAttr getIntZerosAttr(Type type);

/// Utility for generating a constant all ones attribute.
IntegerAttr getIntOnesAttr(Type type);

/// Return the single assignment to a Property value.
PropAssignOp getPropertyAssignment(FIRRTLPropertyValue value);

/// Return the module-scoped driver of a value only looking through one connect.
Value getDriverFromConnect(Value val);

/// Return the value that drives another FIRRTL value within module scope.  This
/// is parameterized by looking through or not through certain constructs.  This
/// assumes a single driver and should only be run after `ExpandWhens`.
Value getModuleScopedDriver(Value val, bool lookThroughWires,
                            bool lookThroughNodes, bool lookThroughCasts);

/// Walk all the drivers of a value, passing in the connect operations drive the
/// value. If the value is an aggregate it will find connects to subfields. If
/// the callback returns false, this function will stop walking.  Returns false
/// if walking was broken, and true otherwise.
using WalkDriverCallback =
    llvm::function_ref<bool(const FieldRef &dst, const FieldRef &src)>;
bool walkDrivers(FIRRTLBaseValue value, bool lookThroughWires,
                 bool lookThroughNodes, bool lookThroughCasts,
                 WalkDriverCallback callback);

/// Get the FieldRef from a value.  This will travel backwards to through the
/// IR, following Subfield and Subindex to find the op which declares the
/// location.
FieldRef getFieldRefFromValue(Value value);

/// Get a string identifier representing the FieldRef.  Return this string and a
/// boolean indicating if a valid "root" for the identifier was found.  If
/// nameSafe is true, this will generate a string that is better suited for
/// naming something in the IR.  E.g., if the fieldRef is a subfield of a
/// subindex, without name safe the output would be:
///
///   foo[42].bar
///
/// With nameSafe, this would be:
///
///   foo_42_bar
std::pair<std::string, bool> getFieldName(const FieldRef &fieldRef,
                                          bool nameSafe = false);

Value getValueByFieldID(ImplicitLocOpBuilder builder, Value value,
                        unsigned fieldID);

/// Walk leaf ground types in the `firrtlType` and apply the function `fn`.
/// The first argument of `fn` is field ID, and the second argument is a
/// leaf ground type.
void walkGroundTypes(FIRRTLType firrtlType,
                     llvm::function_ref<void(uint64_t, FIRRTLBaseType)> fn);
//===----------------------------------------------------------------------===//
// Inner symbol and InnerRef helpers.
//===----------------------------------------------------------------------===//

/// Returns an operation's `inner_sym`, adding one if necessary.
StringAttr
getOrAddInnerSym(Operation *op, StringRef nameHint, FModuleOp mod,
                 std::function<ModuleNamespace &(FModuleOp)> getNamespace);

/// Obtain an inner reference to an operation, possibly adding an `inner_sym`
/// to that operation.
hw::InnerRefAttr
getInnerRefTo(Operation *op, StringRef nameHint,
              std::function<ModuleNamespace &(FModuleOp)> getNamespace);

/// Returns a port's `inner_sym`, adding one if necessary.
StringAttr
getOrAddInnerSym(FModuleLike mod, size_t portIdx, StringRef nameHint,
                 std::function<ModuleNamespace &(FModuleLike)> getNamespace);

/// Obtain an inner reference to a port, possibly adding an `inner_sym`
/// to the port.
hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx, StringRef nameHint,
              std::function<ModuleNamespace &(FModuleLike)> getNamespace);

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

/// If it is a base type, return it as is. If reftype, return wrapped base type.
/// Otherwise, return null.
inline FIRRTLBaseType getBaseType(Type type) {
  return TypeSwitch<Type, FIRRTLBaseType>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base; })
      .Case<RefType>([](auto ref) { return ref.getType(); })
      .Default([](Type type) { return nullptr; });
}

/// Get base type if isa<> the requested type, else null.
template <typename T>
inline T getBaseOfType(Type type) {
  return dyn_cast_or_null<T>(getBaseType(type));
}

/// Return a FIRRTLType with its base type component mutated by the given
/// function. (i.e., ref<T> -> ref<f(T)> and T -> f(T)).
inline FIRRTLType mapBaseType(FIRRTLType type,
                              function_ref<FIRRTLBaseType(FIRRTLBaseType)> fn) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<FIRRTLBaseType>([&](auto base) { return fn(base); })
      .Case<RefType>([&](auto ref) {
        return RefType::get(fn(ref.getType()), ref.getForceable());
      });
}

/// Given a type, return the corresponding lowered type for the HW dialect.
/// Non-FIRRTL types are simply passed through. This returns a null type if it
/// cannot be lowered.
Type lowerType(Type type);

//===----------------------------------------------------------------------===//
// Parser-related utilities
//
// These cannot always be relegated to the parser and sometimes need to be
// available for passes.  This has specifically come up for Annotation lowering
// where there is FIRRTL stuff that needs to be parsed out of an annotation.
//===----------------------------------------------------------------------===//

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, std::optional<mlir::LocationAttr>> maybeStringToLocation(
    StringRef spelling, bool skipParsing, StringAttr &locatorFilenameCache,
    FileLineColLoc &fileLineColLocCache, MLIRContext *context);

//===----------------------------------------------------------------------===//
// Parallel utilities
//===----------------------------------------------------------------------===//

/// Wrapper for llvm::parallelTransformReduce that performs the transform_reduce
/// serially when MLIR multi-threading is disabled.
/// Does not add a ParallelDiagnosticHandler like mlir::parallelFor.
template <class IterTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, IterTy begin, IterTy end,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  // Parallel when enabled
  if (context->isMultithreadingEnabled())
    return llvm::parallelTransformReduce(begin, end, init, reduce, transform);

  // Serial fallback (from llvm::parallelTransformReduce)
  for (IterTy i = begin; i != end; ++i)
    init = reduce(std::move(init), transform(*i));
  return std::move(init);
}

/// Range wrapper
template <class RangeTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, RangeTy &&r,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  return transformReduce(context, std::begin(r), std::end(r), init, reduce,
                         transform);
}

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
