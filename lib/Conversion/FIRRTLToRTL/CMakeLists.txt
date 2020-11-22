add_mlir_conversion_library(MLIRFIRRTLToRTL
  LowerToRTL.cpp
  LowerTypes.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRRTL
  MLIRFIRRTL
  MLIRTransforms
)
