//===- FIRRTLOps.cpp - Implement the FIRRTL operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using mlir::RegionRange;
using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Remove elements at the specified indices from the input array, returning the
/// elements not mentioned.  The indices array is expected to be sorted and
/// unique.
template <typename T>
static SmallVector<T>
removeElementsAtIndices(ArrayRef<T> input, ArrayRef<unsigned> indicesToDrop) {
#ifndef NDEBUG // Check sortedness.
  if (!input.empty()) {
    for (size_t i = 1, e = indicesToDrop.size(); i != e; ++i)
      assert(indicesToDrop[i - 1] < indicesToDrop[i] &&
             "indicesToDrop isn't sorted and unique");
    assert(indicesToDrop.back() < input.size() && "index out of range");
  }
#endif

  // Copy over the live chunks.
  size_t lastCopied = 0;
  SmallVector<T> result;
  result.reserve(input.size() - indicesToDrop.size());

  for (unsigned indexToDrop : indicesToDrop) {
    // If we skipped over some valid elements, copy them over.
    if (indexToDrop > lastCopied) {
      result.append(input.begin() + lastCopied, input.begin() + indexToDrop);
      lastCopied = indexToDrop;
    }
    // Ignore this value so we don't copy it in the next iteration.
    ++lastCopied;
  }

  // If there are live elements at the end, copy them over.
  if (lastCopied < input.size())
    result.append(input.begin() + lastCopied, input.end());

  return result;
}

bool firrtl::isDuplexValue(Value val) {
  Operation *op = val.getDefiningOp();
  // Block arguments are not duplex values.
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
      .Case<SubfieldOp, SubindexOp, SubaccessOp>(
          [](auto op) { return isDuplexValue(op.input()); })
      .Case<RegOp, RegResetOp, WireOp>([](auto) { return true; })
      .Default([](auto) { return false; });
}

Flow firrtl::swapFlow(Flow flow) {
  switch (flow) {
  case Flow::Source:
    return Flow::Sink;
  case Flow::Sink:
    return Flow::Source;
  case Flow::Duplex:
    return Flow::Duplex;
  }
  llvm_unreachable("invalid flow");
}

Flow firrtl::foldFlow(Value val, Flow accumulatedFlow) {
  auto swap = [&accumulatedFlow]() -> Flow {
    return swapFlow(accumulatedFlow);
  };

  if (auto blockArg = val.dyn_cast<BlockArgument>()) {
    auto op = val.getParentBlock()->getParentOp();
    auto direction =
        cast<FModuleLike>(op).getPortDirection(blockArg.getArgNumber());
    if (direction == Direction::Out)
      return swap();
    return accumulatedFlow;
  }

  Operation *op = val.getDefiningOp();

  return TypeSwitch<Operation *, Flow>(op)
      .Case<SubfieldOp>([&](auto op) {
        return foldFlow(op.input(),
                        op.isFieldFlipped() ? swap() : accumulatedFlow);
      })
      .Case<SubindexOp, SubaccessOp>(
          [&](auto op) { return foldFlow(op.input(), accumulatedFlow); })
      // Registers, Wires, and behavioral memory ports are always Duplex.
      .Case<RegOp, RegResetOp, WireOp, MemoryPortOp>(
          [](auto) { return Flow::Duplex; })
      .Case<InstanceOp>([&](auto inst) {
        for (auto arg : llvm::enumerate(inst.getResults()))
          if (arg.value() == val) {
            if (inst.getReferencedModule().getPortDirection(arg.index()) ==
                Direction::Out)
              return accumulatedFlow;
            else
              return swap();
          }
        llvm_unreachable("couldn't find result in results");
      })
      .Case<MemOp>([&](auto op) { return swap(); })
      // Anything else acts like a universal source.
      .Default([&](auto) { return accumulatedFlow; });
}

// TODO: This is doing the same walk as foldFlow.  These two functions can be
// combined and return a (flow, kind) product.
DeclKind firrtl::getDeclarationKind(Value val) {
  Operation *op = val.getDefiningOp();
  if (!op)
    return DeclKind::Port;

  return TypeSwitch<Operation *, DeclKind>(op)
      .Case<InstanceOp>([](auto) { return DeclKind::Instance; })
      .Case<SubfieldOp, SubindexOp, SubaccessOp>(
          [](auto op) { return getDeclarationKind(op.input()); })
      .Default([](auto) { return DeclKind::Other; });
}

size_t firrtl::getNumPorts(Operation *op) {
  if (auto module = dyn_cast<FModuleLike>(op))
    return module.getNumPorts();
  return op->getNumResults();
}

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

void CircuitOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayAttr annotations) {
  // Add an attribute for the name.
  result.addAttribute(builder.getIdentifier("name"), name);

  if (!annotations)
    annotations = builder.getArrayAttr({});
  result.addAttribute("annotations", annotations);

  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

// Return the main module that is the entry point of the circuit.
Operation *CircuitOp::getMainModule() { return lookupSymbol(name()); }

static ParseResult parseCircuitOpAttrs(OpAsmParser &parser,
                                       NamedAttrList &resultAttrs) {
  auto result = parser.parseOptionalAttrDictWithKeyword(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  return result;
}

static void printCircuitOpAttrs(OpAsmPrinter &p, Operation *op,
                                DictionaryAttr attr) {
  // "name" is always elided.
  SmallVector<StringRef> elidedAttrs = {"name"};
  // Elide "annotations" if it doesn't exist or if it is empty
  auto annotationsAttr = op->getAttrOfType<ArrayAttr>("annotations");
  if (annotationsAttr.empty())
    elidedAttrs.push_back("annotations");

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), elidedAttrs);
}

static LogicalResult verifyCircuitOp(CircuitOp circuit) {
  StringRef main = circuit.name();

  // Check that the circuit has a non-empty name.
  if (main.empty()) {
    circuit.emitOpError("must have a non-empty name");
    return failure();
  }

  // Check that a module matching the "main" module exists in the circuit.
  if (!circuit.getMainModule()) {
    circuit.emitOpError("must contain one module that matches main name '" +
                        main + "'");
    return failure();
  }

  // Store a mapping of defname to either the first external module
  // that defines it or, preferentially, the first external module
  // that defines it and has no parameters.
  llvm::DenseMap<Attribute, FExtModuleOp> defnameMap;

  // Verify external modules.
  for (auto &op : *circuit.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(op);
    if (!extModule)
      continue;

    auto defname = extModule.defnameAttr();
    if (!defname)
      continue;

    // Check that this extmodule's defname does not conflict with
    // the symbol name of any module.
    auto collidingModule = circuit.lookupSymbol(defname.getValue());
    if (isa_and_nonnull<FModuleOp>(collidingModule)) {
      auto diag =
          op.emitOpError()
          << "attribute 'defname' with value " << defname
          << " conflicts with the name of another module in the circuit";
      diag.attachNote(collidingModule->getLoc())
          << "previous module declared here";
      return failure();
    }

    // Find an optional extmodule with a defname collision. Update
    // the defnameMap if this is the first extmodule with that
    // defname or if the current extmodule takes no parameters and
    // the collision does. The latter condition improves later
    // extmodule verification as checking against a parameterless
    // module is stricter.
    FExtModuleOp collidingExtModule;
    if (auto &value = defnameMap[defname]) {
      collidingExtModule = value;
      if (value.parameters() && !extModule.parameters())
        value = extModule;
    } else {
      value = extModule;
      // Go to the next extmodule if no extmodule with the same
      // defname was found.
      continue;
    }

    // Check that the number of ports is exactly the same.
    SmallVector<PortInfo> ports = extModule.getPorts();
    SmallVector<PortInfo> collidingPorts = collidingExtModule.getPorts();

    if (ports.size() != collidingPorts.size()) {
      auto diag = op.emitOpError()
                  << "with 'defname' attribute " << defname << " has "
                  << ports.size()
                  << " ports which is different from a previously defined "
                     "extmodule with the same 'defname' which has "
                  << collidingPorts.size() << " ports";
      diag.attachNote(collidingExtModule.getLoc())
          << "previous extmodule definition occurred here";
      return failure();
    }

    // Check that ports match for name and type. Since parameters
    // *might* affect widths, ignore widths if either module has
    // parameters. Note that this allows for misdetections, but
    // has zero false positives.
    for (auto p : llvm::zip(ports, collidingPorts)) {
      StringAttr aName = std::get<0>(p).name, bName = std::get<1>(p).name;
      FIRRTLType aType = std::get<0>(p).type, bType = std::get<1>(p).type;

      if (extModule.parameters() || collidingExtModule.parameters()) {
        aType = aType.getWidthlessType();
        bType = bType.getWidthlessType();
      }
      if (aName != bName) {
        auto diag = op.emitOpError()
                    << "with 'defname' attribute " << defname
                    << " has a port with name " << aName
                    << " which does not match the name of the port "
                    << "in the same position of a previously defined "
                    << "extmodule with the same 'defname', expected port "
                       "to have name "
                    << bName;
        diag.attachNote(collidingExtModule.getLoc())
            << "previous extmodule definition occurred here";
        return failure();
      }
      if (aType != bType) {
        auto diag = op.emitOpError()
                    << "with 'defname' attribute " << defname
                    << " has a port with name " << aName
                    << " which has a different type " << aType
                    << " which does not match the type of the port in "
                       "the same position of a previously defined "
                       "extmodule with the same 'defname', expected port "
                       "to have type "
                    << bType;
        diag.attachNote(collidingExtModule.getLoc())
            << "previous extmodule definition occurred here";
        return failure();
      }
    }
  }

  return success();
}

Region &CircuitOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *CircuitOp::getBody() { return &getBodyRegion().front(); }

//===----------------------------------------------------------------------===//
// FExtModuleOp and FModuleOp
//===----------------------------------------------------------------------===//

/// This function can extract information about ports from a module and an
/// extmodule.
SmallVector<PortInfo> FModuleOp::getPorts() {
  SmallVector<PortInfo> results;
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    results.push_back({getPortNameAttr(i), getPortType(i), getPortDirection(i),
                       getArgument(i).getLoc(),
                       AnnotationSet::forPort(*this, i)});
  }
  return results;
}

/// This function can extract information about ports from a module and an
/// extmodule.
SmallVector<PortInfo> FExtModuleOp::getPorts() {
  // FExtModuleOp's don't have block arguments or locations for their ports.
  auto loc = getLoc();

  SmallVector<PortInfo> results;
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    results.push_back({getPortNameAttr(i), getPortType(i), getPortDirection(i),
                       loc, AnnotationSet::forPort(*this, i)});
  }
  return results;
}

// Return the port with the specified name.
BlockArgument FModuleOp::getArgument(size_t portNumber) {
  return getBody()->getArgument(portNumber);
}

/// Inserts the given ports. The insertion indices are expected to be in order.
/// Insertion occurs in-order, such that ports with the same insertion index
/// appear in the module in the same order they appeared in the list.
void FModuleOp::insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports) {
  if (ports.empty())
    return;
  unsigned oldNumArgs = getNumPorts();
  unsigned newNumArgs = oldNumArgs + ports.size();

  auto *body = getBody();

  // Add direction markers and names for new ports.
  SmallVector<Direction> existingDirections =
      direction::unpackAttribute(this->getPortDirectionsAttr());
  ArrayRef<Attribute> existingNames = this->getPortNames();
  ArrayRef<Attribute> existingTypes = this->getPortTypes();
  assert(existingDirections.size() == oldNumArgs);
  assert(existingNames.size() == oldNumArgs);
  assert(existingTypes.size() == oldNumArgs);

  SmallVector<Direction> newDirections;
  SmallVector<Attribute> newNames;
  SmallVector<Attribute> newTypes;
  newDirections.reserve(newNumArgs);
  newNames.reserve(newNumArgs);
  newTypes.reserve(newNumArgs);

  unsigned oldIdx = 0;
  auto migrateOldPorts = [&](unsigned untilOldIdx) {
    while (oldIdx < oldNumArgs && oldIdx < untilOldIdx) {
      newDirections.push_back(existingDirections[oldIdx]);
      newNames.push_back(existingNames[oldIdx]);
      newTypes.push_back(existingTypes[oldIdx]);
      ++oldIdx;
    }
  };
  for (auto &port : ports) {
    migrateOldPorts(port.first);
    newDirections.push_back(port.second.direction);
    newNames.push_back(port.second.name);
    newTypes.push_back(TypeAttr::get(port.second.type));
    body->insertArgument(port.first, port.second.type, port.second.loc);
  }
  migrateOldPorts(oldNumArgs);

  // Apply these changed markers.
  (*this)->setAttr("portDirections",
                   direction::packAttribute(getContext(), newDirections));
  (*this)->setAttr("portNames", ArrayAttr::get(getContext(), newNames));
  (*this)->setAttr("portTypes", ArrayAttr::get(getContext(), newTypes));
}

/// Erases the ports listed in `portIndices`.  `portIndices` is expected to
/// be in order and unique.
void FModuleOp::erasePorts(ArrayRef<unsigned> portIndices) {
  if (portIndices.empty())
    return;

  // Drop the direction markers for dead ports.
  SmallVector<Direction> directions =
      direction::unpackAttribute(this->getPortDirectionsAttr());
  ArrayRef<Attribute> portNames = this->getPortNames();
  ArrayRef<Attribute> portAnno = this->getPortAnnotations();
  ArrayRef<Attribute> portTypes = this->getPortTypes();
  assert(directions.size() == portNames.size());

  SmallVector<Direction> newDirections =
      removeElementsAtIndices<Direction>(directions, portIndices);
  SmallVector<Attribute> newPortNames =
      removeElementsAtIndices(portNames, portIndices);
  SmallVector<Attribute> newPortAnno =
      removeElementsAtIndices(portAnno, portIndices);
  SmallVector<Attribute> newPortTypes =
      removeElementsAtIndices(portTypes, portIndices);
  (*this)->setAttr("portDirections",
                   direction::packAttribute(getContext(), newDirections));
  (*this)->setAttr("portNames", ArrayAttr::get(getContext(), newPortNames));
  (*this)->setAttr("portAnnotations",
                   ArrayAttr::get(getContext(), newPortAnno));
  (*this)->setAttr("portTypes", ArrayAttr::get(getContext(), newPortTypes));

  // Erase the block arguments.
  getBody()->eraseArguments(portIndices);
}

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<PortInfo> ports,
                        ArrayAttr annotations) {
  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  // Record the names of the arguments if present.
  SmallVector<Direction, 4> portDirections;
  SmallVector<Attribute, 4> portNames;
  SmallVector<Attribute, 4> portTypes;
  SmallVector<Attribute, 4> portAnnotations;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    portDirections.push_back(ports[i].direction);
    portNames.push_back(ports[i].name);
    portTypes.push_back(TypeAttr::get(ports[i].type));
    portAnnotations.push_back(ports[i].annotations.getArrayAttr());
  }

  // Both attributes are added, even if the module has no ports.
  result.addAttribute(
      "portDirections",
      direction::packAttribute(builder.getContext(), portDirections));
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("portTypes", builder.getArrayAttr(portTypes));
  result.addAttribute("portAnnotations", builder.getArrayAttr(portAnnotations));

  if (!annotations)
    annotations = builder.getArrayAttr({});
  result.addAttribute("annotations", annotations);

  result.addRegion();
}

void FModuleOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayRef<PortInfo> ports,
                      ArrayAttr annotations) {
  buildModule(builder, result, name, ports, annotations);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.type, elt.loc);
}

void FExtModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, ArrayRef<PortInfo> ports,
                         StringRef defnameAttr, ArrayAttr annotations) {
  buildModule(builder, result, name, ports, annotations);
  if (!defnameAttr.empty())
    result.addAttribute("defname", builder.getStringAttr(defnameAttr));
}

/// Print a module signature in the following form:
///   in x: !firrtl.uint<1> [{class = "DontTouch}], out "_port": !firrtl.uint<2>
///
/// When there is no block specified, the port names print as MLIR identifiers,
/// wrapping in quotes if not legal to print as-is. When there is no block
/// specified, this function always return false, indicating that there was no
/// issue printing port names.
///
/// If there is a block specified, then port names will be printed as SSA
/// values.  If there is a reason the printed SSA values can't match the true
/// port name, then this function will return true.  When this happens, the
/// caller should print the port names as a part of the `attr-dict`.
static bool printModuleSignature(OpAsmPrinter &p, Block *block,
                                 ArrayRef<Direction> portDirections,
                                 ArrayRef<Attribute> portNames,
                                 ArrayRef<Attribute> portTypes,
                                 ArrayRef<Attribute> portAnnotations) {
  // When printing port names as SSA values, we can fail to print them
  // identically.
  bool printedNamesDontMatch = false;

  // If we are printing the ports as block arguments the op must have a first
  // block.
  SmallString<32> resultNameStr;
  p << '(';
  for (unsigned i = 0, e = portTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    // Print the port direction.
    p << portDirections[i] << " ";

    // Print the port name.  If there is a valid block, we print it as a block
    // argument.
    if (block) {
      // Get the printed format for the argument name.
      resultNameStr.clear();
      llvm::raw_svector_ostream tmpStream(resultNameStr);
      p.printOperand(block->getArgument(i), tmpStream);
      // If the name wasn't printable in a way that agreed with portName, make
      // sure to print out an explicit portNames attribute.
      auto portName = portNames[i].cast<StringAttr>().getValue();
      if (!portName.empty() && tmpStream.str().drop_front() != portName)
        printedNamesDontMatch = true;
      p << tmpStream.str();
    } else {
      p.printIdentifier(portNames[i].cast<StringAttr>().getValue());
    }

    // Print the port type.
    p << ": ";
    auto portType = portTypes[i].cast<TypeAttr>().getValue();
    p.printType(portType);

    // Print the port specific annotations. The port annotations array will be
    // empty if there are none.
    if (!portAnnotations.empty() &&
        !portAnnotations[i].cast<ArrayAttr>().empty()) {
      p << " ";
      p.printAttribute(portAnnotations[i]);
    }
  }

  p << ')';
  return printedNamesDontMatch;
}

/// Parse a module signature.  If port names are SSA identifiers, then this will
/// populate `entryArgs`.
static ParseResult
parseModuleSignature(OpAsmParser &parser, bool hasSSAIdentifiers,
                     SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
                     SmallVectorImpl<Direction> &portDirections,
                     SmallVectorImpl<Attribute> &portNames,
                     SmallVectorImpl<Attribute> &portTypes,
                     SmallVectorImpl<Attribute> &portAnnotations) {
  auto *context = parser.getContext();

  auto parseArgument = [&]() -> ParseResult {
    // Parse port direction.
    if (succeeded(parser.parseOptionalKeyword("out")))
      portDirections.push_back(Direction::Out);
    else if (succeeded(parser.parseKeyword("in", "or 'out'")))
      portDirections.push_back(Direction::In);
    else
      return failure();

    // Parse the port name.
    if (hasSSAIdentifiers) {
      OpAsmParser::OperandType arg;
      if (parser.parseRegionArgument(arg))
        return failure();
      entryArgs.push_back(arg);
      // The name of an argument is of the form "%42" or "%id", and since
      // parsing succeeded, we know it always has one character.
      assert(arg.name.size() > 1 && arg.name[0] == '%' && "Unknown MLIR name");
      if (isdigit(arg.name[1]))
        portNames.push_back(StringAttr::get(context, ""));
      else
        portNames.push_back(StringAttr::get(context, arg.name.drop_front()));
    } else {
      StringAttr portName;
      if (parser.parseIdentifier(portName))
        return failure();
      portNames.push_back(portName);
    }

    // Parse the port type.
    Type portType;
    if (parser.parseColonType(portType))
      return failure();
    portTypes.push_back(TypeAttr::get(portType));

    // Parse any port annotations. TODO: The API for parsing optional attributes
    // is missing some functions and we shouldn't need to create a dummy
    // attribute list. x-ref TODO in SVOps.cpp, parseOmitEmptyStringAttr.
    ArrayAttr annos;
    NamedAttrList dummy;
    parser.parseOptionalAttribute(annos, /*type=*/{}, "dummy", dummy);
    if (!annos)
      annos = parser.getBuilder().getArrayAttr({});
    portAnnotations.push_back(annos);

    return success();
  };

  // Parse all ports.
  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseArgument);
}

static void printFModuleLikeOp(OpAsmPrinter &p, FModuleLike op) {
  // Print the operation and the function name.
  p << " ";
  p.printSymbolName(op.moduleName());

  // Both modules and external modules have a body, but it is always empty for
  // external modules.
  Block *body = nullptr;
  if (!op->getRegion(0).empty())
    body = &op->getRegion(0).front();

  auto portDirections = direction::unpackAttribute(op.getPortDirectionsAttr());

  auto needPortNamesAttr =
      printModuleSignature(p, body, portDirections, op.getPortNames(),
                           op.getPortTypes(), op.getPortAnnotations());

  SmallVector<StringRef, 4> omittedAttrs = {"sym_name", "portDirections",
                                            "portTypes", "portAnnotations"};

  // We can omit the portNames if they were able to be printed as properly as
  // block arguments.
  if (!needPortNamesAttr)
    omittedAttrs.push_back("portNames");

  // If there are no annotations we can omit the empty array.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    omittedAttrs.push_back("annotations");

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), omittedAttrs);
}

static void printFExtModuleOp(OpAsmPrinter &p, FExtModuleOp op) {
  printFModuleLikeOp(p, op);
}

static void printFModuleOp(OpAsmPrinter &p, FModuleOp op) {
  printFModuleLikeOp(p, op);

  // Print the body if this is not an external function. Since this block does
  // not have terminators, printing the terminator actually just prints the last
  // operation.
  Region &body = op.body();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

static ParseResult parseFModuleLikeOp(OpAsmParser &parser,
                                      OperationState &result,
                                      bool hasSSAIdentifiers) {
  auto *context = result.getContext();
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::OperandType> entryArgs;
  SmallVector<Direction, 4> portDirections;
  SmallVector<Attribute, 4> portNames;
  SmallVector<Attribute, 4> portTypes;
  SmallVector<Attribute, 4> portAnnotations;
  if (parseModuleSignature(parser, hasSSAIdentifiers, entryArgs, portDirections,
                           portNames, portTypes, portAnnotations))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(portNames.size() == portTypes.size());

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.

  // Add port directions.
  if (!result.attributes.get("portDirections"))
    result.addAttribute("portDirections",
                        direction::packAttribute(context, portDirections));

  // Add port names.
  if (!result.attributes.get("portNames")) {
    result.addAttribute("portNames", builder.getArrayAttr(portNames));
  }

  // Add the port types.
  if (!result.attributes.get("portTypes"))
    result.addAttribute("portTypes", ArrayAttr::get(context, portTypes));

  // Add the port annotations.
  if (!result.attributes.get("portAnnotations")) {
    // If there are no portAnnotations, don't add the attribute.
    if (llvm::any_of(portAnnotations, [&](Attribute anno) {
          return !anno.cast<ArrayAttr>().empty();
        }))
      result.addAttribute("portAnnotations",
                          ArrayAttr::get(context, portAnnotations));
  }

  // The annotations attribute is always present, but not printed when empty.
  if (!result.attributes.get("annotations"))
    result.addAttribute("annotations", builder.getArrayAttr({}));

  // The portAnnotations attribute is always present, but not printed when
  // empty.
  if (!result.attributes.get("portAnnotations"))
    result.addAttribute("portAnnotations", builder.getArrayAttr({}));

  // Parse the optional function body.
  auto *body = result.addRegion();

  if (hasSSAIdentifiers) {
    // Collect block argument types.
    SmallVector<Type, 4> argTypes;
    if (!entryArgs.empty())
      llvm::transform(portTypes, std::back_inserter(argTypes),
                      [](Attribute typeAttr) -> Type {
                        return typeAttr.cast<TypeAttr>().getValue();
                      });

    if (parser.parseRegion(*body, entryArgs, argTypes))
      return failure();
    if (body->empty())
      body->push_back(new Block());
  }
  return success();
}

static ParseResult parseFModuleOp(OpAsmParser &parser, OperationState &result) {
  return parseFModuleLikeOp(parser, result, /*hasSSAIdentifiers=*/true);
}

static ParseResult parseFExtModuleOp(OpAsmParser &parser,
                                     OperationState &result) {
  return parseFModuleLikeOp(parser, result, /*hasSSAIdentifiers=*/false);
}

static LogicalResult verifyFExtModuleOp(FExtModuleOp op) {
  auto paramDictOpt = op.parameters();
  if (!paramDictOpt)
    return success();

  DictionaryAttr paramDict = paramDictOpt.getValue();
  auto checkParmValue = [&](NamedAttribute elt) -> bool {
    auto value = elt.second;
    if (value.isa<IntegerAttr>() || value.isa<StringAttr>() ||
        value.isa<FloatAttr>())
      return true;
    op.emitError() << "has unknown extmodule parameter value '" << elt.first
                   << "' = " << value;
    return false;
  };

  if (!llvm::all_of(paramDict, checkParmValue))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
FModuleLike InstanceOp::getReferencedModule() {
  auto circuit = (*this)->getParentOfType<CircuitOp>();
  if (!circuit)
    return nullptr;

  return circuit.lookupSymbol<FModuleLike>(moduleNameAttr());
}

FModuleLike InstanceOp::getReferencedModule(SymbolTable &symbolTable) {
  return symbolTable.lookup<FModuleLike>(moduleNameAttr().getLeafReference());
}

void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       TypeRange resultTypes, StringRef moduleName,
                       StringRef name, ArrayRef<Attribute> annotations,
                       ArrayRef<Attribute> portAnnotations, bool lowerToBind) {
  result.addAttribute("moduleName",
                      SymbolRefAttr::get(builder.getContext(), moduleName));
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("annotations", builder.getArrayAttr(annotations));
  result.addAttribute("lowerToBind", builder.getBoolAttr(lowerToBind));
  result.addTypes(resultTypes);

  if (portAnnotations.empty()) {
    SmallVector<Attribute, 16> portAnnotationsVec(resultTypes.size(),
                                                  builder.getArrayAttr({}));
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotationsVec));
  } else {
    assert(portAnnotations.size() == resultTypes.size());
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotations));
  }
}

ArrayAttr InstanceOp::getPortAnnotation(unsigned portIdx) {
  assert(portIdx < getNumResults() &&
         "index should be smaller than result number");
  return portAnnotations()[portIdx].cast<ArrayAttr>();
}

void InstanceOp::setAllPortAnnotations(ArrayRef<Attribute> annotations) {
  assert(annotations.size() == getNumResults() &&
         "number of annotations is not equal to result number");
  (*this)->setAttr("portAnnotations",
                   ArrayAttr::get(getContext(), annotations));
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<FModuleOp>();
  auto referencedModule =
      symbolTable.lookupNearestSymbolFrom<FModuleLike>(*this, moduleNameAttr());
  if (!referencedModule) {
    emitOpError("invalid symbol reference");
    return failure();
  }

  // Check that this instance doesn't recursively instantiate its wrapping
  // module.
  if (referencedModule == module) {
    auto diag = emitOpError()
                << "is a recursive instantiation of its containing module";
    diag.attachNote(module.getLoc()) << "containing module declared here";
    return failure();
  }

  SmallVector<PortInfo> modulePorts = referencedModule.getPorts();

  // Check that result types are consistent with the referenced module's ports.
  size_t numResults = getNumResults();
  if (numResults != modulePorts.size()) {
    auto diag = emitOpError()
                << "has a wrong number of results; expected "
                << modulePorts.size() << " but got " << numResults;
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";
    return failure();
  }

  for (size_t i = 0; i != numResults; i++) {
    auto resultType = getResult(i).getType();
    auto expectedType = modulePorts[i].type;
    if (resultType != expectedType) {
      auto diag = emitOpError()
                  << "result type for " << modulePorts[i].name << " must be "
                  << expectedType << ", but got " << resultType;

      diag.attachNote(referencedModule->getLoc())
          << "original module declared here";
      return failure();
    }
  }

  return success();
}

/// Verify the correctness of an InstanceOp.
static LogicalResult verifyInstanceOp(InstanceOp instance) {

  // Check that this instance is inside a module.
  auto module = instance->getParentOfType<FModuleOp>();
  if (!module) {
    instance.emitOpError("should be embedded in a 'firrtl.module'");
    return failure();
  }

  if (instance.portAnnotations().size() != instance.getNumResults())
    return instance.emitOpError("the number of result annotations should be "
                                "equal to the number of results");

  return success();
}

void MemoryPortOp::build(OpBuilder &builder, OperationState &result,
                         Type dataType, Value memory, MemDirAttr direction,
                         StringRef name, ArrayRef<Attribute> annotations) {
  build(builder, result, CMemoryPortType::get(builder.getContext()), dataType,
        memory, direction, name, builder.getArrayAttr(annotations));
}

LogicalResult MemoryPortOp::inferReturnTypes(MLIRContext *context,
                                             Optional<Location> loc,
                                             ValueRange operands,
                                             DictionaryAttr attrs,
                                             mlir::RegionRange regions,
                                             SmallVectorImpl<Type> &results) {
  auto inType = operands[0].getType();
  auto memType = inType.dyn_cast<CMemoryType>();
  if (!memType) {
    if (loc)
      mlir::emitError(*loc, "memory port requires memory operand");
    return failure();
  }
  results.push_back(memType.getElementType());
  results.push_back(CMemoryPortType::get(context));
  return success();
}

static LogicalResult verifyMemoryPortOp(MemoryPortOp memoryPort) {
  // MemoryPorts require exactly 1 access. Right now there are no other
  // operations that could be using that value due to the types.
  if (!memoryPort.port().hasOneUse())
    return memoryPort.emitOpError(
        "port should be used by a firrtl.memoryport.access");
  return success();
}

MemoryPortAccessOp MemoryPortOp::getAccess() {
  auto uses = port().use_begin();
  if (uses == port().use_end())
    return {};
  return cast<MemoryPortAccessOp>(uses->getOwner());
}

void MemOp::build(OpBuilder &builder, OperationState &result,
                  TypeRange resultTypes, uint32_t readLatency,
                  uint32_t writeLatency, uint64_t depth, RUWAttr ruw,
                  ArrayRef<Attribute> portNames, StringRef name,
                  ArrayRef<Attribute> annotations,
                  ArrayRef<Attribute> portAnnotations) {
  result.addAttribute(
      "readLatency",
      builder.getIntegerAttr(builder.getIntegerType(32), readLatency));
  result.addAttribute(
      "writeLatency",
      builder.getIntegerAttr(builder.getIntegerType(32), writeLatency));
  result.addAttribute(
      "depth", builder.getIntegerAttr(builder.getIntegerType(64), depth));
  result.addAttribute("ruw", ::RUWAttrAttr::get(builder.getContext(), ruw));
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("annotations", builder.getArrayAttr(annotations));
  result.addTypes(resultTypes);

  if (portAnnotations.empty()) {
    SmallVector<Attribute, 16> portAnnotationsVec(resultTypes.size(),
                                                  builder.getArrayAttr({}));
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotationsVec));
  } else {
    assert(portAnnotations.size() == resultTypes.size());
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotations));
  }
}

ArrayAttr MemOp::getPortAnnotation(unsigned portIdx) {
  assert(portIdx < getNumResults() &&
         "index should be smaller than result number");
  return portAnnotations()[portIdx].cast<ArrayAttr>();
}

void MemOp::setAllPortAnnotations(ArrayRef<Attribute> annotations) {
  assert(annotations.size() == getNumResults() &&
         "number of annotations is not equal to result number");
  (*this)->setAttr("portAnnotations",
                   ArrayAttr::get(getContext(), annotations));
}

// Get the number of read, write and read-write ports.
void MemOp::getNumPorts(size_t &numReadPorts, size_t &numWritePorts,
                        size_t &numReadWritePorts) {
  numReadPorts = 0;
  numWritePorts = 0;
  numReadWritePorts = 0;
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto portKind = getPortKind(i);
    if (portKind == MemOp::PortKind::Read)
      ++numReadPorts;
    else if (portKind == MemOp::PortKind::Write) {
      ++numWritePorts;
    } else
      ++numReadWritePorts;
  }
}

/// Verify the correctness of a MemOp.
static LogicalResult verifyMemOp(MemOp mem) {

  // Store the port names as we find them. This lets us check quickly
  // for uniqueneess.
  llvm::SmallDenseSet<Attribute, 8> portNamesSet;

  // Store the previous data type. This lets us check that the data
  // type is consistent across all ports.
  FIRRTLType oldDataType;

  for (size_t i = 0, e = mem.getNumResults(); i != e; ++i) {
    auto portName = mem.getPortName(i);

    // Get a bundle type representing this port, stripping an outer
    // flip if it exists.  If this is not a bundle<> or
    // flip<bundle<>>, then this is an error.
    BundleType portBundleType =
        TypeSwitch<FIRRTLType, BundleType>(
            mem.getResult(i).getType().cast<FIRRTLType>())
            .Case<BundleType>([](BundleType a) { return a; })
            .Default([](auto) { return nullptr; });
    if (!portBundleType) {
      mem.emitOpError() << "has an invalid type on port " << portName
                        << " (expected '!firrtl.bundle<...>')";
      return failure();
    }

    // Require that all port names are unique.
    if (!portNamesSet.insert(portName).second) {
      mem.emitOpError() << "has non-unique port name " << portName;
      return failure();
    }

    // Determine the kind of the memory.  If the kind cannot be
    // determined, then it's indicative of the wrong number of fields
    // in the type (but we don't know any more just yet).
    MemOp::PortKind portKind;
    {
      auto elt = mem.getPortNamed(portName);
      if (!elt) {
        mem.emitOpError() << "could not get port with name " << portName;
        return failure();
      }
      auto firrtlType = elt.getType().cast<FIRRTLType>();
      auto portType = firrtlType.dyn_cast<BundleType>();
      switch (portType.getNumElements()) {
      case 4:
        portKind = MemOp::PortKind::Read;
        break;
      case 5:
        portKind = MemOp::PortKind::Write;
        break;
      case 7:
        portKind = MemOp::PortKind::ReadWrite;
        break;
      default:
        mem.emitOpError()
            << "has an invalid number of fields on port " << portName
            << " (expected 4 for read, 5 for write, or 7 for read/write)";
        return failure();
      }
    }

    // Safely search for the "data" field, erroring if it can't be
    // found.
    FIRRTLType dataType;
    {
      auto dataTypeOption = portBundleType.getElement("data");
      if (!dataTypeOption && portKind == MemOp::PortKind::ReadWrite)
        dataTypeOption = portBundleType.getElement("wdata");
      if (!dataTypeOption) {
        mem.emitOpError() << "has no data field on port " << portName
                          << " (expected to see \"data\" for a read or write "
                             "port or \"rdata\" for a read/write port)";
        return failure();
      }
      dataType = dataTypeOption.getValue().type;
      // Read data is expected to ba a flip.
      if (portKind == MemOp::PortKind::Read) {
        // FIXME error on missing bundle flip
      }
    }

    // Error if the data type isn't passive.
    if (!dataType.isPassive()) {
      mem.emitOpError() << "has non-passive data type on port " << portName
                        << " (memory types must be passive)";
      return failure();
    }

    // Error if the data type contains analog types.
    if (dataType.containsAnalog()) {
      mem.emitOpError()
          << "has a data type that contains an analog type on port " << portName
          << " (memory types cannot contain analog types)";
      return failure();
    }

    // Check that the port type matches the kind that we determined
    // for this port.  This catches situations of extraneous port
    // fields beind included or the fields being named incorrectly.
    FIRRTLType expectedType =
        mem.getTypeForPort(mem.depth(), dataType, portKind,
                           dataType.isGround() ? mem.getMaskBits() : 0);
    // Compute the original port type as portBundleType may have
    // stripped outer flip information.
    auto originalType = mem.getResult(i).getType();
    if (originalType != expectedType) {
      StringRef portKindName;
      switch (portKind) {
      case MemOp::PortKind::Read:
        portKindName = "read";
        break;
      case MemOp::PortKind::Write:
        portKindName = "write";
        break;
      case MemOp::PortKind::ReadWrite:
        portKindName = "readwrite";
        break;
      }
      mem.emitOpError() << "has an invalid type for port " << portName
                        << " of determined kind \"" << portKindName
                        << "\" (expected " << expectedType << ", but got "
                        << originalType << ")";
      return failure();
    }

    // Error if the type of the current port was not the same as the
    // last port, but skip checking the first port.
    if (oldDataType && oldDataType != dataType) {
      mem.emitOpError() << "port " << mem.getPortName(i)
                        << " has a different type than port "
                        << mem.getPortName(i - 1) << " (expected "
                        << oldDataType << ", but got " << dataType << ")";
      return failure();
    }

    oldDataType = dataType;
  }

  auto maskWidth = mem.getMaskBits();

  auto dataWidth = mem.getDataType().getBitWidthOrSentinel();
  if (dataWidth > 0 && maskWidth > (size_t)dataWidth)
    return mem.emitOpError("the mask width cannot be greater than "
                           "data width");

  if (mem.portAnnotations().size() != mem.getNumResults())
    return mem.emitOpError("the number of result annotations should be "
                           "equal to the number of results");

  return success();
}

BundleType MemOp::getTypeForPort(uint64_t depth, FIRRTLType dataType,
                                 PortKind portKind, size_t maskBits) {

  auto *context = dataType.getContext();
  FIRRTLType maskType;
  // maskBits not specified (==0), then get the mask type from the dataType.
  if (maskBits == 0)
    maskType = dataType.getMaskType();
  else
    maskType = UIntType::get(context, maskBits);

  auto getId = [&](StringRef name) -> StringAttr {
    return StringAttr::get(context, name);
  };

  SmallVector<BundleType::BundleElement, 7> portFields;

  auto addressType =
      UIntType::get(context, std::max(1U, llvm::Log2_64_Ceil(depth)));

  portFields.push_back({getId("addr"), false, addressType});
  portFields.push_back({getId("en"), false, UIntType::get(context, 1)});
  portFields.push_back({getId("clk"), false, ClockType::get(context)});

  switch (portKind) {
  case PortKind::Read:
    portFields.push_back({getId("data"), true, dataType});
    break;

  case PortKind::Write:
    portFields.push_back({getId("data"), false, dataType});
    portFields.push_back({getId("mask"), false, maskType});
    break;

  case PortKind::ReadWrite:
    portFields.push_back({getId("rdata"), true, dataType});
    portFields.push_back({getId("wmode"), false, UIntType::get(context, 1)});
    portFields.push_back({getId("wdata"), false, dataType});
    portFields.push_back({getId("wmask"), false, maskType});
    break;
  }

  return BundleType::get(portFields, context).cast<BundleType>();
}

/// Return the kind of port this is given the port type from a 'mem' decl.
static MemOp::PortKind getMemPortKindFromType(FIRRTLType type) {
  auto portType = type.dyn_cast<BundleType>();
  switch (portType.getNumElements()) {
  case 4:
    return MemOp::PortKind::Read;
  case 5:
    return MemOp::PortKind::Write;
  default:
    return MemOp::PortKind::ReadWrite;
  }
}

/// Return the name and kind of ports supported by this memory.
SmallVector<MemOp::NamedPort> MemOp::getPorts() {
  SmallVector<MemOp::NamedPort> result;
  // Each entry in the bundle is a port.
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    // Each port is a bundle.
    auto portType = getResult(i).getType().cast<FIRRTLType>();
    result.push_back({getPortName(i), getMemPortKindFromType(portType)});
  }
  return result;
}

/// Return the kind of the specified port.
MemOp::PortKind MemOp::getPortKind(StringRef portName) {
  return getMemPortKindFromType(
      getPortNamed(portName).getType().cast<FIRRTLType>());
}

/// Return the kind of the specified port number.
MemOp::PortKind MemOp::getPortKind(size_t resultNo) {
  return getMemPortKindFromType(
      getResult(resultNo).getType().cast<FIRRTLType>());
}

/// Return the number of bits in the mask for the memory.
size_t MemOp::getMaskBits() {

  for (auto res : getResults()) {
    auto firstPortType = res.getType().cast<FIRRTLType>();
    if (getMemPortKindFromType(firstPortType) == PortKind::Read)
      continue;

    FIRRTLType mType;
    for (auto t :
         firstPortType.getPassiveType().cast<BundleType>().getElements()) {
      if (t.name.getValue().contains("mask"))
        mType = t.type;
    }
    if (mType.dyn_cast_or_null<UIntType>())
      return mType.getBitWidthOrSentinel();
  }
  // Mask of zero bits means, either there are no write/readwrite ports or the
  // mask is of aggregate type.
  return 0;
}

/// Return the data-type field of the memory, the type of each element.
FIRRTLType MemOp::getDataType() {
  assert(getNumResults() != 0 && "Mems with no read/write ports are illegal");

  auto firstPortType = getResult(0).getType().cast<FIRRTLType>();

  StringRef dataFieldName = "data";
  if (getMemPortKindFromType(firstPortType) == PortKind::ReadWrite)
    dataFieldName = "rdata";

  return firstPortType.getPassiveType().cast<BundleType>().getElementType(
      dataFieldName);
}

StringAttr MemOp::getPortName(size_t resultNo) {
  return portNames()[resultNo].cast<StringAttr>();
}

FIRRTLType MemOp::getPortType(size_t resultNo) {
  return results()[resultNo].getType().cast<FIRRTLType>();
}

Value MemOp::getPortNamed(StringAttr name) {
  auto namesArray = portNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i] == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

/// Infer the return types of this operation.
LogicalResult NodeOp::inferReturnTypes(MLIRContext *context,
                                       Optional<Location> loc,
                                       ValueRange operands,
                                       DictionaryAttr attrs,
                                       mlir::RegionRange regions,
                                       SmallVectorImpl<Type> &results) {
  results.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

static LogicalResult verifyConnectOp(ConnectOp connect) {
  FIRRTLType destType = connect.dest().getType().cast<FIRRTLType>();
  FIRRTLType srcType = connect.src().getType().cast<FIRRTLType>();

  // Analog types cannot be connected and must be attached.
  if (destType.isa<AnalogType>() || srcType.isa<AnalogType>())
    return connect.emitError("analog types may not be connected");
  if (auto destBundle = destType.dyn_cast<BundleType>())
    if (destBundle.containsAnalog())
      return connect.emitError("analog types may not be connected");
  if (auto srcBundle = srcType.dyn_cast<BundleType>())
    if (srcBundle.containsAnalog())
      return connect.emitError("analog types may not be connected");

  // Destination and source types must be equivalent.
  if (!areTypesEquivalent(destType, srcType))
    return connect.emitError("type mismatch between destination ")
           << destType << " and source " << srcType;

  // Destination bitwidth must be greater than or equal to source bitwidth.
  int32_t destWidth = destType.getPassiveType().getBitWidthOrSentinel();
  int32_t srcWidth = srcType.getPassiveType().getBitWidthOrSentinel();
  if (destWidth > -1 && srcWidth > -1 && destWidth < srcWidth)
    return connect.emitError("destination width ")
           << destWidth << " is not greater than or equal to source width "
           << srcWidth;

  // TODO: Relax this to allow reads from output ports,
  // instance/memory input ports.
  if (foldFlow(connect.src()) == Flow::Sink) {
    // A sink that is a port output or instance input used as a source is okay.
    auto kind = getDeclarationKind(connect.src());
    if (kind != DeclKind::Port && kind != DeclKind::Instance) {
      auto diag =
          connect.emitOpError()
          << "has invalid flow: the right-hand-side has sink flow and "
             "is not an output port or instance input (expected source "
             "flow, duplex flow, an output port, or an instance input).";
      return diag.attachNote(connect.src().getLoc())
             << "the right-hand-side was defined here.";
    }
  }

  if (foldFlow(connect.dest()) == Flow::Source) {
    auto diag = connect.emitOpError()
                << "has invalid flow: the left-hand-side has source flow "
                   "(expected sink or duplex flow).";
    return diag.attachNote(connect.dest().getLoc())
           << "the left-hand-side was defined here.";
  }

  return success();
}

static LogicalResult verifyPartialConnectOp(PartialConnectOp partialConnect) {
  FIRRTLType destType = partialConnect.dest().getType().cast<FIRRTLType>();
  FIRRTLType srcType = partialConnect.src().getType().cast<FIRRTLType>();

  if (!areTypesWeaklyEquivalent(destType, srcType))
    return partialConnect.emitError("type mismatch between destination ")
           << destType << " and source " << srcType
           << ". Types are not weakly equivalent.";

  // Check that the flows make sense.
  if (foldFlow(partialConnect.src()) == Flow::Sink) {
    // A sink that is a port output or instance input used as a source is okay.
    auto kind = getDeclarationKind(partialConnect.src());
    if (kind != DeclKind::Port && kind != DeclKind::Instance) {
      auto diag =
          partialConnect.emitOpError()
          << "has invalid flow: the right-hand-side has sink flow and "
             "is not an output port or instance input (expected source "
             "flow, duplex flow, an output port, or an instance input).";
      return diag.attachNote(partialConnect.src().getLoc())
             << "the right-hand-side was defined here.";
    }
  }

  if (foldFlow(partialConnect.dest()) == Flow::Source) {
    auto diag = partialConnect.emitOpError()
                << "has invalid flow: the left-hand-side has source flow "
                   "(expected sink or duplex flow).";
    return diag.attachNote(partialConnect.dest().getLoc())
           << "the left-hand-side was defined here.";
  }

  return success();
}

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  elseRegion().push_back(new Block());
}

void WhenOp::build(OpBuilder &builder, OperationState &result, Value condition,
                   bool withElseRegion, std::function<void()> thenCtor,
                   std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);
  result.addOperands(condition);

  // Create "then" region.
  builder.createBlock(result.addRegion());
  if (thenCtor)
    thenCtor();

  // Create "else" region.
  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    builder.createBlock(elseRegion);
    if (elseCtor)
      elseCtor();
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// Type inference adaptor that narrows from the very generic MLIR
/// `InferTypeOpInterface` to what we need in the FIRRTL dialect: just operands
/// and attributes, no context or regions. Also, we only ever produce a single
/// result value, so the FIRRTL-specific type inference ops directly return the
/// inferred type rather than pushing into the `results` vector.
LogicalResult impl::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &results,
    llvm::function_ref<FIRRTLType(ValueRange, ArrayRef<NamedAttribute>,
                                  Optional<Location>)>
        callback) {
  auto type = callback(
      operands, attrs ? attrs.getValue() : ArrayRef<NamedAttribute>{}, loc);
  if (type) {
    results.push_back(type);
    return success();
  }
  return failure();
}

/// Get an attribute by name from a list of named attributes. Aborts if the
/// attribute does not exist.
static Attribute getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  for (auto attr : attrs)
    if (attr.first == name)
      return attr.second;
  llvm::report_fatal_error("attribute '" + name + "' not found");
}

/// Same as above, but casts the attribute to a specific type.
template <typename AttrClass>
AttrClass getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  return getAttr(attrs, name).cast<AttrClass>();
}

/// Return true if the specified operation is a firrtl expression.
bool firrtl::isExpression(Operation *op) {
  struct IsExprClassifier : public ExprVisitor<IsExprClassifier, bool> {
    bool visitInvalidExpr(Operation *op) { return false; }
    bool visitUnhandledExpr(Operation *op) { return true; }
  };

  return IsExprClassifier().dispatchExprVisitor(op);
}

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << " ";
  p.printAttributeWithoutType(op.valueAttr());
  p << " : ";
  p.printType(op.getType());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  // Parse the constant value, without knowing its width.
  APInt value;
  auto loc = parser.getCurrentLocation();
  auto valueResult = parser.parseOptionalInteger(value);
  if (!valueResult.hasValue())
    return parser.emitError(loc, "expected integer value");

  // Parse the result firrtl integer type.
  IntType resultType;
  if (failed(*valueResult) || parser.parseColonType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(resultType);

  // Now that we know the width and sign of the result type, we can munge the
  // APInt as appropriate.
  if (resultType.hasWidth()) {
    auto width = (unsigned)resultType.getWidthOrSentinel();
    if (width == 0)
      return parser.emitError(loc, "zero bit constants aren't allowed");

    if (width > value.getBitWidth()) {
      // sext is always safe here, even for unsigned values, because the
      // parseOptionalInteger method will return something with a zero in the
      // top bits if it is a positive number.
      value = value.sext(width);
    } else if (width < value.getBitWidth()) {
      // The parser can return an unnecessarily wide result with leading zeros.
      // This isn't a problem, but truncating off bits is bad.
      if (value.getNumSignBits() < value.getBitWidth() - width)
        return parser.emitError(loc, "constant too large for result type ")
               << resultType;
      value = value.trunc(width);
    }
  }

  auto intType = parser.getBuilder().getIntegerType(value.getBitWidth(),
                                                    resultType.isSigned());
  auto valueAttr = parser.getBuilder().getIntegerAttr(intType, value);
  result.addAttribute("value", valueAttr);
  return success();
}

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = constant.getType().cast<IntType>();
  auto width = intType.getWidthOrSentinel();
  if (width != -1 && (int)constant.value().getBitWidth() != width)
    return constant.emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");

  // The sign of the attribute's integer type must match our integer type sign.
  auto attrType = constant.valueAttr().getType().cast<IntegerType>();
  if (attrType.isSignless() ||
      attrType.isSigned() != constant.getType().isSigned())
    return constant.emitError("firrtl.constant attribute has wrong sign");

  return success();
}

/// Build a ConstantOp from an APInt and a FIRRTL type, handling the attribute
/// formation for the 'value' attribute.
void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       const APInt &value) {
  int32_t width = type.getWidthOrSentinel();
  (void)width;
  assert((width == -1 || (int32_t)value.getBitWidth() == width) &&
         "incorrect attribute bitwidth for firrtl.constant");

  auto attr =
      IntegerAttr::get(type.getContext(), APSInt(value, !type.isSigned()));
  return build(builder, result, type, attr);
}

/// Build a ConstantOp from an APSInt, handling the attribute formation for the
/// 'value' attribute and inferring the FIRRTL type.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APSInt &value) {
  auto attr = IntegerAttr::get(builder.getContext(), value);
  auto type =
      IntType::get(builder.getContext(), value.isSigned(), value.getBitWidth());
  return build(builder, result, type, attr);
}

static void printSpecialConstantOp(OpAsmPrinter &p, SpecialConstantOp &op) {
  p << " ";
  // SpecialConstant uses a BoolAttr, and we want to print `true` as `1`.
  p << static_cast<unsigned>(op.value());
  p << " : ";
  p.printType(op.getType());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
}

static ParseResult parseSpecialConstantOp(OpAsmParser &parser,
                                          OperationState &result) {
  // Parse the constant value.  SpecialConstant uses bool attributes, but it
  // prints as an integer.
  APInt value;
  auto loc = parser.getCurrentLocation();
  auto valueResult = parser.parseOptionalInteger(value);
  if (!valueResult.hasValue())
    return parser.emitError(loc, "expected integer value");

  // Clocks and resets can only be 0 or 1.
  if (value != 0 && value != 1)
    return parser.emitError(loc, "special constants can only be 0 or 1.");

  // Parse the result firrtl type.
  Type resultType;
  if (failed(*valueResult) || parser.parseColonType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(resultType);

  // Create the attribute.
  auto valueAttr = parser.getBuilder().getBoolAttr(value == 1);
  result.addAttribute("value", valueAttr);
  return success();
}

static LogicalResult verifySubfieldOp(SubfieldOp op) {
  if (op.fieldIndex() >=
      op.input().getType().cast<BundleType>().getNumElements())
    return op.emitOpError("subfield element index is greater than the number "
                          "of fields in the bundle type");
  return success();
}

/// Return true if the specified operation has a constant value. This trivially
/// checks for `firrtl.constant` and friends, but also looks through subaccesses
/// and correctly handles wires driven with only constant values.
bool firrtl::isConstant(Operation *op) {
  // Worklist of ops that need to be examined that should all be constant in
  // order for the input operation to be constant.
  SmallVector<Operation *, 8> worklist({op});

  // Mutable state indicating if this op is a constant.  Assume it is a constant
  // and look for counterexamples.
  bool constant = true;

  // While we haven't found a counterexample and there are still ops in the
  // worklist, pull ops off the worklist.  If it provides a counterexample, set
  // the `constant` to false (and exit on the next loop iteration).  Otherwise,
  // look through the op or spawn off more ops to look at.
  while (constant && !(worklist.empty()))
    TypeSwitch<Operation *>(worklist.pop_back_val())
        .Case<NodeOp, AsSIntPrimOp, AsUIntPrimOp>([&](auto op) {
          if (auto definingOp = op.input().getDefiningOp())
            worklist.push_back(definingOp);
          constant = false;
        })
        .Case<WireOp, SubindexOp, SubfieldOp>([&](auto op) {
          for (auto &use : op.getResult().getUses())
            worklist.push_back(use.getOwner());
        })
        .Case<ConstantOp, SpecialConstantOp>([](auto) {})
        .Default([&](auto) { constant = false; });

  return constant;
}

/// Return true if the specified value is a constant. This trivially checks for
/// `firrtl.constant` and friends, but also looks through subaccesses and
/// correctly handles wires driven with only constant values.
bool firrtl::isConstant(Value value) {
  if (auto *op = value.getDefiningOp())
    return isConstant(op);
  return false;
}

FIRRTLType SubfieldOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Optional<Location> loc) {
  auto inType = operands[0].getType().cast<BundleType>();
  auto fieldIndex =
      getAttr<IntegerAttr>(attrs, "fieldIndex").getValue().getZExtValue();

  if (fieldIndex >= inType.getNumElements()) {
    if (loc)
      mlir::emitError(*loc, "subfield element index is greater than the number "
                            "of fields in the bundle type");
    return {};
  }

  // SubfieldOp verifier checks that the field index is valid with number of
  // subelements.
  return inType.getElement(fieldIndex).type;
}

bool SubfieldOp::isFieldFlipped() {
  auto bundle = input().getType().cast<BundleType>();
  return bundle.getElement(fieldIndex()).isFlip;
}

FIRRTLType SubindexOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Optional<Location> loc) {
  auto inType = operands[0].getType();
  auto fieldIdx =
      getAttr<IntegerAttr>(attrs, "index").getValue().getZExtValue();

  if (auto vectorType = inType.dyn_cast<FVectorType>()) {
    if (fieldIdx < vectorType.getNumElements())
      return vectorType.getElementType();
    if (loc)
      mlir::emitError(*loc, "out of range index '")
          << fieldIdx << "' in vector type " << inType;
    return {};
  }

  if (loc)
    mlir::emitError(*loc, "subindex requires vector operand");
  return {};
}

FIRRTLType SubaccessOp::inferReturnType(ValueRange operands,
                                        ArrayRef<NamedAttribute> attrs,
                                        Optional<Location> loc) {
  auto inType = operands[0].getType();
  auto indexType = operands[1].getType();

  if (!indexType.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc, "subaccess index must be UInt type, not ")
          << indexType;
    return {};
  }

  if (auto vectorType = inType.dyn_cast<FVectorType>())
    return vectorType.getElementType();

  if (loc)
    mlir::emitError(*loc, "subaccess requires vector operand, not ") << inType;
  return {};
}

//===----------------------------------------------------------------------===//
// Binary Primitives
//===----------------------------------------------------------------------===//

/// If LHS and RHS are both UInt or SInt types, the return true and fill in the
/// width of them if known.  If unknown, return -1 for the widths.
///
/// On failure, this reports and error and returns false.  This function should
/// not be used if you don't want an error reported.
static bool isSameIntTypeKind(Type lhs, Type rhs, int32_t &lhsWidth,
                              int32_t &rhsWidth, Optional<Location> loc) {
  // Must have two integer types with the same signedness.
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsi = rhs.dyn_cast<IntType>();
  if (!lhsi || !rhsi || lhsi.isSigned() != rhsi.isSigned()) {
    if (loc) {
      if (lhsi && !rhsi)
        mlir::emitError(*loc, "second operand must be an integer type, not ")
            << rhs;
      else if (!lhsi && rhsi)
        mlir::emitError(*loc, "first operand must be an integer type, not ")
            << lhs;
      else if (!lhsi && !rhsi)
        mlir::emitError(*loc, "operands must be integer types, not ")
            << lhs << " and " << rhs;
      else
        mlir::emitError(*loc, "operand signedness must match");
    }
    return false;
  }

  lhsWidth = lhsi.getWidthOrSentinel();
  rhsWidth = rhsi.getWidthOrSentinel();
  return true;
}

LogicalResult impl::verifySameOperandsIntTypeKind(Operation *op) {
  assert(op->getNumOperands() == 2 &&
         "SameOperandsIntTypeKind on non-binary op");
  int32_t lhsWidth, rhsWidth;
  return success(isSameIntTypeKind(op->getOperand(0).getType(),
                                   op->getOperand(1).getType(), lhsWidth,
                                   rhsWidth, op->getLoc()));
}

LogicalResult impl::validateBinaryOpArguments(ValueRange operands,
                                              ArrayRef<NamedAttribute> attrs,
                                              Location loc) {
  if (operands.size() != 2 || !attrs.empty()) {
    mlir::emitError(loc, "operation requires two operands and no constants");
    return failure();
  }
  return success();
}

FIRRTLType impl::inferAddSubResult(FIRRTLType lhs, FIRRTLType rhs,
                                   Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth) + 1;
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType MulPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;

  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType DivPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  // For unsigned, the width is the width of the numerator on the LHS.
  if (lhs.isa<UIntType>())
    return UIntType::get(lhs.getContext(), lhsWidth);

  // For signed, the width is the width of the numerator on the LHS, plus 1.
  int32_t resultWidth = lhsWidth != -1 ? lhsWidth + 1 : -1;
  return SIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType RemPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::min(lhsWidth, rhsWidth);
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType impl::inferBitwiseResult(FIRRTLType lhs, FIRRTLType rhs,
                                    Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth);
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType impl::inferComparisonResult(FIRRTLType lhs, FIRRTLType rhs,
                                       Optional<Location> loc) {
  return UIntType::get(lhs.getContext(), 1);
}

FIRRTLType CatPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            Optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType DShlPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                             Optional<Location> loc) {
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsui = rhs.dyn_cast<UIntType>();
  if (!rhsui || !lhsi) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }

  // If the left or right has unknown result type, then the operation does
  // too.
  auto width = lhsi.getWidthOrSentinel();
  if (width == -1 || !rhsui.getWidth().hasValue()) {
    width = -1;
  } else {
    auto amount = rhsui.getWidth().getValue();
    if (amount >= 32) {
      if (loc)
        mlir::emitError(*loc, "shift amount too large: second operand of dshl "
                              "is wider than 31 bits");
      return {};
    }
    int64_t newWidth = (int64_t)width + ((int64_t)1 << amount) - 1;
    if (newWidth > INT32_MAX) {
      if (loc)
        mlir::emitError(*loc, "shift amount too large: first operand shifted "
                              "by maximum amount exceeds maximum width");
      return {};
    }
    width = newWidth;
  }
  return IntType::get(lhs.getContext(), lhsi.isSigned(), width);
}

FIRRTLType DShlwPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                              Optional<Location> loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

FIRRTLType DShrPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                             Optional<Location> loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

//===----------------------------------------------------------------------===//
// Unary Primitives
//===----------------------------------------------------------------------===//

LogicalResult impl::validateUnaryOpArguments(ValueRange operands,
                                             ArrayRef<NamedAttribute> attrs,
                                             Location loc) {
  if (operands.size() != 1 || !attrs.empty()) {
    mlir::emitError(loc, "operation requires one operand and no constants");
    return failure();
  }
  return success();
}

FIRRTLType AsSIntPrimOp::inferUnaryReturnType(FIRRTLType input,
                                              Optional<Location> loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar type");
    return {};
  }
  return SIntType::get(input.getContext(), width);
}

FIRRTLType AsUIntPrimOp::inferUnaryReturnType(FIRRTLType input,
                                              Optional<Location> loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar type");
    return {};
  }
  return UIntType::get(input.getContext(), width);
}

FIRRTLType AsAsyncResetPrimOp::inferUnaryReturnType(FIRRTLType input,
                                                    Optional<Location> loc) {
  int32_t width = input.getBitWidthOrSentinel();
  if (width == -2 || width == 0 || width > 1) {
    if (loc)
      mlir::emitError(*loc, "operand must be single bit scalar type");
    return {};
  }
  return AsyncResetType::get(input.getContext());
}

FIRRTLType AsClockPrimOp::inferUnaryReturnType(FIRRTLType input,
                                               Optional<Location> loc) {
  return ClockType::get(input.getContext());
}

FIRRTLType CvtPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           Optional<Location> loc) {
  if (auto uiType = input.dyn_cast<UIntType>()) {
    auto width = uiType.getWidthOrSentinel();
    if (width != -1)
      ++width;
    return SIntType::get(input.getContext(), width);
  }

  if (input.isa<SIntType>())
    return input;

  if (loc)
    mlir::emitError(*loc, "operand must have integer type");
  return {};
}

FIRRTLType NegPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           Optional<Location> loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "operand must have integer type");

    return {};
  }
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    ++width;
  return SIntType::get(input.getContext(), width);
}

FIRRTLType NotPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           Optional<Location> loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "operand must have integer type");

    return {};
  }
  return UIntType::get(input.getContext(), inputi.getWidthOrSentinel());
}

FIRRTLType impl::inferReductionResult(FIRRTLType input,
                                      Optional<Location> loc) {
  return UIntType::get(input.getContext(), 1);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult BitsPrimOp::validateArguments(ValueRange operands,
                                            ArrayRef<NamedAttribute> attrs,
                                            Location loc) {
  if (operands.size() != 1 || attrs.size() != 2) {
    mlir::emitError(loc, "operation requires one operand and two constants");
    return failure();
  }
  return success();
}

FIRRTLType BitsPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Optional<Location> loc) {
  auto input = operands[0].getType();
  auto high = getAttr<IntegerAttr>(attrs, "hi").getValue().getSExtValue();
  auto low = getAttr<IntegerAttr>(attrs, "lo").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "input type should be the int type but got ")
          << input;
    return {};
  }

  // High must be >= low and both most be non-negative.
  if (high < low) {
    if (loc)
      mlir::emitError(*loc,
                      "high must be equal or greater than low, but got high = ")
          << high << ", low = " << low;
    return {};
  }

  if (low < 0) {
    if (loc)
      mlir::emitError(*loc, "low must be non-negative but got ") << low;
    return {};
  }

  // If the input has staticly known width, check it.  Both and low must be
  // strictly less than width.
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && high >= width) {
    if (loc)
      mlir::emitError(*loc)
          << "high must be smaller than the width of input, but got high = "
          << high << ", width = " << width;
    return {};
  }

  return UIntType::get(input.getContext(), high - low + 1);
}

LogicalResult impl::validateOneOperandOneConst(ValueRange operands,
                                               ArrayRef<NamedAttribute> attrs,
                                               Location loc) {
  if (operands.size() != 1 || attrs.size() != 1) {
    mlir::emitError(loc, "operation requires one operand and one constant");
    return failure();
  }
  return success();
}

FIRRTLType HeadPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "operand must have integer type and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && amount > width) {
    if (loc)
      mlir::emitError(*loc, "amount larger than input width");
    return {};
  }

  width = std::max<int32_t>(width, amount);
  return UIntType::get(input.getContext(), amount);
}

LogicalResult MuxPrimOp::validateArguments(ValueRange operands,
                                           ArrayRef<NamedAttribute> attrs,
                                           Location loc) {
  if (operands.size() != 3 || attrs.size() != 0) {
    mlir::emitError(loc, "operation requires three operands and no constants");
    return failure();
  }
  return success();
}

FIRRTLType MuxPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
  auto high = operands[1].getType().cast<FIRRTLType>();
  auto low = operands[2].getType().cast<FIRRTLType>();

  // TODO: Should use a more general type equivalence operator.
  if (high == low)
    return low;

  // The base types need to be equivalent.
  if (high.getTypeID() != low.getTypeID()) {
    if (loc)
      mlir::emitError(*loc, "true and false value must have same type");
    return {};
  }

  if (low.isa<IntType>()) {
    // Two different Int types can be compatible.  If either has unknown
    // width, then return it.  If both are known but different width, then
    // return the larger one.
    int32_t highWidth = high.getBitWidthOrSentinel();
    int32_t lowWidth = low.getBitWidthOrSentinel();
    if (lowWidth == -1)
      return low;
    if (highWidth == -1)
      return high;
    return lowWidth > highWidth ? low : high;
  }

  // FIXME: Should handle bundles and other things.
  if (loc)
    mlir::emitError(*loc, "unknown types to mux");
  return {};
}

FIRRTLType PadPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "pad input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width == -1)
    return inputi;

  width = std::max<int32_t>(width, amount);
  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShlPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "shl input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width += amount;

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShrPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "shr input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width = std::max<int32_t>(1, width - amount);

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType TailPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "tail input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1) {
    if (width < amount) {
      if (loc)
        mlir::emitError(*loc,
                        "amount must be less than or equal operand width");
      return {};
    }
    width -= amount;
  }

  return IntType::get(input.getContext(), false, width);
}

//===----------------------------------------------------------------------===//
// VerbatimExprOp
//===----------------------------------------------------------------------===//

void VerbatimExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the text is macro like, then use a pretty name.  We only take the
  // text up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = text();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// VerbatimWireOp
//===----------------------------------------------------------------------===//

void VerbatimWireOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the text is macro like, then use a pretty name.  We only take the
  // text up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = text();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// Conversions to/from structs in the standard dialect.
//===----------------------------------------------------------------------===//

static LogicalResult verifyHWStructCastOp(HWStructCastOp cast) {
  // We must have a bundle and a struct, with matching pairwise fields
  BundleType bundleType;
  hw::StructType structType;
  if ((bundleType = cast.getOperand().getType().dyn_cast<BundleType>())) {
    structType = cast.getType().dyn_cast<hw::StructType>();
    if (!structType)
      return cast.emitError("result type must be a struct");
  } else if ((bundleType = cast.getType().dyn_cast<BundleType>())) {
    structType = cast.getOperand().getType().dyn_cast<hw::StructType>();
    if (!structType)
      return cast.emitError("operand type must be a struct");
  } else {
    return cast.emitError("either source or result type must be a bundle type");
  }

  auto firFields = bundleType.getElements();
  auto hwFields = structType.getElements();
  if (firFields.size() != hwFields.size())
    return cast.emitError("bundle and struct have different number of fields");

  for (size_t findex = 0, fend = firFields.size(); findex < fend; ++findex) {
    if (firFields[findex].name.getValue() != hwFields[findex].name)
      return cast.emitError("field names don't match '")
             << firFields[findex].name.getValue() << "', '"
             << hwFields[findex].name << "'";
    int64_t firWidth =
        FIRRTLType(firFields[findex].type).getBitWidthOrSentinel();
    int64_t hwWidth = hw::getBitWidth(hwFields[findex].type);
    if (firWidth > 0 && hwWidth > 0 && firWidth != hwWidth)
      return cast.emitError("size of field '")
             << hwFields[findex].name << "' don't match " << firWidth << ", "
             << hwWidth;
  }

  return success();
}

static LogicalResult verifyBitCastOp(BitCastOp cast) {

  auto inTypeBits = getBitWidth(cast.getOperand().getType().cast<FIRRTLType>());
  auto resTypeBits = getBitWidth(cast.getType());
  if (inTypeBits.hasValue() && resTypeBits.hasValue()) {
    // Bitwidths must match for valid bitcast.
    if (inTypeBits.getValue() == resTypeBits.getValue())
      return success();
    return cast.emitError("the bitwidth of input (")
           << inTypeBits.getValue() << ") and result ("
           << resTypeBits.getValue() << ") don't match";
  }
  if (!inTypeBits.hasValue())
    return cast.emitError(
               "bitwidth cannot be determined for input operand type ")
           << cast.getOperand().getType();
  return cast.emitError("bitwidth cannot be determined for result type ")
         << cast.getType();
}

//===----------------------------------------------------------------------===//
// Custom attr-dict Directive that Elides Annotations
//===----------------------------------------------------------------------===//

/// Parse an optional attribute dictionary, adding an empty 'annotations'
/// attribute if not specified.
static ParseResult parseElideAnnotations(OpAsmParser &parser,
                                         NamedAttrList &resultAttrs) {
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  return result;
}

static void printElideAnnotations(OpAsmPrinter &p, Operation *op,
                                  DictionaryAttr attr,
                                  ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef> elidedAttrs(extraElides.begin(), extraElides.end());
  // Elide "annotations" if it is empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elidedAttrs.push_back("annotations");

  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

/// Parse an optional attribute dictionary, adding empty 'annotations' and
/// 'portAnnotations' attributes if not specified.
static ParseResult parseElidePortAnnotations(OpAsmParser &parser,
                                             NamedAttrList &resultAttrs) {
  auto result = parseElideAnnotations(parser, resultAttrs);

  if (!resultAttrs.get("portAnnotations")) {
    SmallVector<Attribute, 16> portAnnotations(
        parser.getNumResults(), parser.getBuilder().getArrayAttr({}));
    resultAttrs.append("portAnnotations",
                       parser.getBuilder().getArrayAttr(portAnnotations));
  }
  return result;
}

// Elide 'annotations' and 'portAnnotations' attributes if they are empty.
static void printElidePortAnnotations(OpAsmPrinter &p, Operation *op,
                                      DictionaryAttr attr,
                                      ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef, 2> elidedAttrs(extraElides.begin(), extraElides.end());

  if (llvm::all_of(op->getAttrOfType<ArrayAttr>("portAnnotations"),
                   [&](Attribute a) { return a.cast<ArrayAttr>().empty(); }))
    elidedAttrs.push_back("portAnnotations");
  printElideAnnotations(p, op, attr, elidedAttrs);
}

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &resultAttrs) {

  if (parseElideAnnotations(parser, resultAttrs))
    return failure();

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  if (resultAttrs.get("name"))
    return success();

  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({Identifier::get("name", context), nameAttr});
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 DictionaryAttr attr,
                                 ArrayRef<StringRef> extraElides = {}) {
  // List of attributes to elide when printing the dictionary.
  SmallVector<StringRef, 2> elides(extraElides.begin(), extraElides.end());

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    elides.push_back("name");

  printElideAnnotations(p, op, attr, elides);
}

//===----------------------------------------------------------------------===//
// InstanceOp Custom attr-dict Directive
//===----------------------------------------------------------------------===//

static ParseResult parseInstanceOp(OpAsmParser &parser,
                                   NamedAttrList &resultAttrs) {
  auto result = parseElidePortAnnotations(parser, resultAttrs);

  if (!resultAttrs.get("lowerToBind")) {
    resultAttrs.append("lowerToBind", parser.getBuilder().getBoolAttr(false));
  }

  return result;
}

/// Always elide "moduleName", elide "lowerToBind" if false, and elide
/// "annotations" if it exists or if it is empty.
static void printInstanceOp(OpAsmPrinter &p, Operation *op,
                            DictionaryAttr attr) {
  SmallVector<StringRef, 2> elides = {"moduleName"};
  if (auto lowerToBind = op->getAttrOfType<BoolAttr>("lowerToBind"))
    if (!lowerToBind.getValue())
      elides.push_back("lowerToBind");

  printElidePortAnnotations(p, op, attr, elides);
}

//===----------------------------------------------------------------------===//
// MemoryPortOp Custom attr-dict Directive
//===----------------------------------------------------------------------===//

void MemoryPortOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  StringRef base = name();
  if (base.empty())
    base = "memport";
  setNameFn(data(), (base + "_data").str());
  setNameFn(port(), (base + "_port").str());
}

static ParseResult parseMemoryPortOp(OpAsmParser &parser,
                                     NamedAttrList &resultAttrs) {
  return parseElideAnnotations(parser, resultAttrs);
}

/// Always elide "direction" and elide "annotations" if it exists or
/// if it is empty.
static void printMemoryPortOp(OpAsmPrinter &p, Operation *op,
                              DictionaryAttr attr) {
  // "direction" is always elided.
  printElideAnnotations(p, op, attr, {"direction"});
}

//===----------------------------------------------------------------------===//
// SeqMemOp Custom attr-dict Directive
//===----------------------------------------------------------------------===//

static ParseResult parseSeqMemOp(OpAsmParser &parser,
                                 NamedAttrList &resultAttrs) {
  return parseImplicitSSAName(parser, resultAttrs);
}

/// Always elide "ruw" and elide "annotations" if it exists or if it is empty.
static void printSeqMemOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr) {
  printImplicitSSAName(p, op, attr, {"ruw"});
}

//===----------------------------------------------------------------------===//
// MemOp Custom attr-dict Directive
//===----------------------------------------------------------------------===//

static ParseResult parseMemOp(OpAsmParser &parser, NamedAttrList &resultAttrs) {
  return parseElidePortAnnotations(parser, resultAttrs);
}

/// Always elide "ruw" and elide "annotations" if it exists or if it is empty.
static void printMemOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr) {
  // "ruw" is always elided.
  printElidePortAnnotations(p, op, attr, {"ruw"});
}

//===----------------------------------------------------------------------===//
// Miscellaneous custom elision logic.
//===----------------------------------------------------------------------===//

static ParseResult parseElideEmptyName(OpAsmParser &p,
                                       NamedAttrList &resultAttrs) {
  auto result = p.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("name"))
    resultAttrs.append("name", p.getBuilder().getStringAttr(""));

  return result;
}

static void printElideEmptyName(OpAsmPrinter &p, Operation *op,
                                DictionaryAttr attr,
                                ArrayRef<StringRef> extraElides = {}) {

  SmallVector<StringRef> elides(extraElides.begin(), extraElides.end());
  if (op->getAttrOfType<StringAttr>("name").getValue().empty())
    elides.push_back("name");

  p.printOptionalAttrDict(op->getAttrs(), elides);
}

static ParseResult parsePrintfAttrs(OpAsmParser &p,
                                    NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printPrintfAttrs(OpAsmPrinter &p, Operation *op,
                             DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"formatString"});
}

static ParseResult parseStopAttrs(OpAsmParser &p, NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printStopAttrs(OpAsmPrinter &p, Operation *op,
                           DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"exitCode"});
}

static ParseResult parseVerifAttrs(OpAsmParser &p, NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printVerifAttrs(OpAsmPrinter &p, Operation *op,
                            DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"message"});
}

//===----------------------------------------------------------------------===//
// TblGen Generated Logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
