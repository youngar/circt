//===- FIRParser.h - .fir to FIRRTL dialect parser --------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .fir file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRPARSER_H
#define CIRCT_DIALECT_FIRRTL_FIRPARSER_H

#include "circt/Support/LLVM.h"

namespace llvm {
class SourceMgr;
}

namespace mlir {
class LocationAttr;
}

namespace circt {
namespace firrtl {

struct FIRParserOptions {
  /// If this is set to true, the @info locators are ignored, and the locations
  /// are set to the location in the .fir file.
  bool ignoreInfoLocators = false;
  /// If this is set to true, the annotations are just attached to the circuit
  /// and not scattered or processed.
  bool rawAnnotations = false;
  /// The number of annotation files that were specified on the command line.
  /// This, along with numOMIRFiles provides structure to the buffers in the
  /// source manager.
  unsigned numAnnotationFiles;
  /// If this is set to true, the parser will immediately recursively expand any
  /// bulk connects.
  bool expandConnects = false;
  /// If true, then the parser will NOT generate debug taps for "named" wires
  /// and nodes.  A "named" wire/node is one whose name does NOT beging with a
  /// leading "_".  If false, debug-ability is greatly increased.  If true, much
  /// more compact Verilog will be generated.
  bool disableNamePreservation = true;
};

mlir::OwningOpRef<mlir::ModuleOp> importFIRFile(llvm::SourceMgr &sourceMgr,
                                                mlir::MLIRContext *context,
                                                FIRParserOptions options = {});

// Decode a source locator string `spelling`, returning a pair indicating that
// the the `spelling` was correct and an optional location attribute.  The
// `skipParsing` option can be used to short-circuit parsing and just do
// validation of the `spelling`.  This require both an Identifier and a
// FileLineColLoc to use for caching purposes and context as the cache may be
// updated with a new identifier.
//
// This utility exists because source locators can exist outside of normal
// "parsing".  E.g., these can show up in annotations or in Object Model 2.0
// JSON.
//
// TODO: This API is super wacky and should be streamlined to hide the
// caching.
std::pair<bool, llvm::Optional<mlir::LocationAttr>>
maybeStringToLocation(llvm::StringRef spelling, bool skipParsing,
                      mlir::StringAttr &locatorFilenameCache,
                      FileLineColLoc &fileLineColLocCache,
                      MLIRContext *context);

void registerFromFIRFileTranslation();

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRPARSER_H
