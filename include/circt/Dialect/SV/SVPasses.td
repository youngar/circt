//===-- SVPasses.td - SV pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES
#define CIRCT_DIALECT_SV_SVPASSES

include "mlir/Pass/PassBase.td"

def AlwaysFusion : Pass<"sv-always-fusion", "mlir::ModuleOp"> {
  let summary = "Combine conditional blocks with the same predicate";
  let description = [{
      TODO
  }];

  let constructor = "circt::sv::createAlwaysFusionPass()";
}

#endif // CIRCT_DIALECT_SV_SVPASSES
