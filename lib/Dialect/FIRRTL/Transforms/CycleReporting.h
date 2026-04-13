//===- CycleReporting.h - Common Cycle Reporting Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for reporting cycles in FIRRTL modules.
// Used by both CheckCombLoops and CheckDomainCycles passes.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEREPORTING_H
#define LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEREPORTING_H

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace firrtl {

/// Report a cycle in a FIRRTL module with a standardized error message format.
/// This is a common utility used by both CheckCombLoops and CheckDomainCycles.
///
/// Both passes work with FieldRef-based paths, so this function operates
/// directly on FieldRef arrays rather than using templates.
///
/// \param moduleLoc The location of the module containing the cycle
/// \param moduleName The name of the module
/// \param cycleType Description of the cycle type (e.g., "combinational", "domain")
/// \param path The sequence of FieldRef nodes forming the cycle
static void reportCycle(mlir::Location moduleLoc, llvm::StringRef moduleName,
                        llvm::StringRef cycleType,
                        llvm::ArrayRef<FieldRef> path) {
  auto errorDiag =
      mlir::emitError(moduleLoc, "detected ")
      << cycleType << " cycle in a FIRRTL module";

  // Find a named value in the cycle for better error messages
  std::string firstName;
  const FieldRef *namedNode = nullptr;
  for (const auto &node : path) {
    auto [name, rootKnown] = getFieldName(node);
    if (!name.empty()) {
      if (firstName.empty() || name < firstName) {
        firstName = name;
        namedNode = &node;
      }
    }
  }

  if (!namedNode) {
    errorDiag.append(", but unable to find names for any involved values.");
    return;
  }

  // Build the cycle path message starting from the named node
  errorDiag.append(", sample path: ");
  errorDiag << moduleName << ".{" << firstName;

  // Print the cycle starting from the named node
  auto startIt = llvm::find_if(path, [&](const FieldRef &n) {
    return &n == namedNode;
  });

  bool lastWasDots = false;
  for (const auto &node : llvm::concat<const FieldRef>(
           llvm::make_range(std::next(startIt), path.end()),
           llvm::make_range(path.begin(), std::next(startIt)))) {
    auto [name, rootKnown] = getFieldName(node);
    if (!name.empty()) {
      errorDiag << " <- " << name;
      lastWasDots = false;
    } else {
      if (!lastWasDots)
        errorDiag << " <- ...";
      lastWasDots = true;
    }
  }
  errorDiag << "}";
}

} // namespace firrtl
} // namespace circt

#endif // LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEREPORTING_H
