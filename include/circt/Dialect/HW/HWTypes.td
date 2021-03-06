//===- HWTypes.td - HW data type definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the HW dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class HWType<string name> : TypeDef<HWDialect, name> { }

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWIntegerType : DialectType<HWDialect, CPred<"isHWIntegerType($_self)">,
                                 "an integer bitvector of one or more bits",
                                 "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : DialectType<HWDialect, CPred<"isHWValueType($_self)">,
                               "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : DialectType<HWDialect, CPred<"!hasHWInOutType($_self)">,
                                  "a type without inout">;

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

// A simple fixed size array.
def ArrayType : HWType<"Array"> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized HW arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "array";
  let parameters = (ins "::mlir::Type":$elementType, "size_t":$size);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ArrayType get(Type elementType, size_t size) {
      return get(elementType.getContext(), elementType, size);
    }
  }];
}

// An 'unpacked' array of fixed size.
def UnpackedArrayType : HWType<"UnpackedArray"> {
  let summary = "SystemVerilog 'unpacked' fixed-sized array";
  let description = [{
    Unpacked arrays are a more flexible array representation than packed arrays,
    and are typically used to model memories.  See SystemVerilog Spec 7.4.2.
  }];

  let mnemonic = "uarray";
  let parameters = (ins "::mlir::Type":$elementType, "size_t":$size);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static UnpackedArrayType get(Type elementType, size_t size) {
      return get(elementType.getContext(), elementType, size);
    }
  }];
}

def InOutType : HWType<"InOut"> {
  let summary = "inout type";
  let description = [{
    InOut type is used for model operations and values that have "connection"
    semantics, instead of typical dataflow behavior.  This is used for wires
    and inout ports in Verilog.
  }];

  let mnemonic = "inout";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static InOutType get(Type elementType) {
      return get(elementType.getContext(), elementType);
    }
  }];
}

def StructType : HWType<"Struct"> {
  let summary = "HW struct type";
  let description = [{
    Represents a structure of name, value pairs.
    !hw.struct<fieldName1: Type1, fieldName2: Type2>
  }];
  let mnemonic = "struct";

  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::circt::hw::StructType::FieldInfo",
      "struct fields">: $elements
  );

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::hw::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type>&);
  }];
}

def UnionType : HWType<"Union"> {
  let summary = "An untagged union of types";
  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::circt::hw::UnionType::FieldInfo",
      "union fields">: $elements
  );
  let mnemonic = "union";

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::hw::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
  }];
}

def TypeAliasType : HWType<"TypeAlias"> {
  let summary = "An symbolic reference to a type declaration";
  let description = [{
    A TypeAlias is parameterized by a SymbolRefAttr, which points to a
    TypedeclOp. The root reference should refer to a TypeScope within the same
    outer ModuleOp, and the leaf reference should refer to a type within that
    TypeScope. A TypeAlias is further parameterized by the inner type, which is
    needed to be known at the time the type is parsed.

    Upon construction, a TypeAlias stores the symbol reference and type, and
    canonicalizes the type to resolve any nested type aliases. The canonical
    type is also cached to avoid recomputing it when needed.
  }];

  let mnemonic = "typealias";

  let parameters = (ins
    "mlir::SymbolRefAttr":$ref,
    "mlir::Type":$innerType,
    "mlir::Type":$canonicalType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::SymbolRefAttr":$ref, "mlir::Type":$innerType)>
  ];

  let extraClassDeclaration = [{
    Optional<TypedeclOp> getDecl(Operation *op);
  }];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

/// An attribute to indicate the output file an operation should be emitted to.
def OutputFileAttr : StructAttr<"OutputFileAttr", HWDialect, [
  StructFieldAttr<"directory", OptionalAttr<StrAttr>>,
  StructFieldAttr<"name", OptionalAttr<StrAttr>>,
  StructFieldAttr<"exclude_from_filelist",
    DefaultValuedAttr<BoolAttr, "false">>,
  StructFieldAttr<"exclude_replicated_ops",
    DefaultValuedAttr<BoolAttr, "true">>,
]>;
