//===- InitAllPasses.h - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLPASSES_H_
#define CIRCT_INITALLPASSES_H_

#include "circt/Conversion/FIRRTLToLLHD/FIRRTLToLLHD.h"
#include "circt/Conversion/HandshakeToFIRRTL/HandshakeToFIRRTL.h"
#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Conversion/StandardToHandshake/StandardToHandshake.h"
#include "circt/Conversion/StandardToStaticLogic/StandardToStaticLogic.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/SV/SVPasses.h"

namespace circt {

inline void registerAllConversionPasses() {
  registerConversionPasses();
  handshake::registerHandshakeToFIRRTLPasses();
  handshake::registerStandardToHandshakePasses();
  llhd::initLLHDToLLVMPass();
  llhd::registerFIRRTLToLLHDPasses();
  llhd::registerRTLToLLHDPasses();
  staticlogic::registerStandardToStaticLogicPasses();
}

inline void registerAllPasses() {
  // Conversion Passes
  registerAllConversionPasses();

  // Standard Passes
  esi::registerESIPasses();
  firrtl::registerPasses();
  llhd::initLLHDTransformationPasses();
  sv::registerPasses();
}

} // namespace circt

#endif // CIRCT_INITALLPASSES_H_
