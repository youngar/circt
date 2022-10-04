//===- FIRAnnotations.cpp - FIRRTL Annotation Utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with FIRRTL Annotations.
//
//===----------------------------------------------------------------------===//

#include "FIRAnnotations.h"

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;

#define DEBUG_TYPE "lower-annos"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;
using mlir::UnitAttr;

/// Convert arbitrary JSON to an MLIR Attribute.
/// NOLINTNEXTLINE(misc-no-recursion)
static Attribute convertJSONToAttribute(MLIRContext *context,
                                        json::Value &value, json::Path p) {
  // String or quoted JSON
  if (auto a = value.getAsString()) {
    // Test to see if this might be quoted JSON (a string that is actually
    // JSON).  Sometimes FIRRTL developers will do this to serialize objects
    // that the Scala FIRRTL Compiler doesn't know about.
    auto unquotedValue = json::parse(*a);
    auto err = unquotedValue.takeError();
    // If this parsed without an error and we didn't just unquote a number, then
    // it's more JSON and recurse on that.
    //
    // We intentionally do not want to unquote a number as, in JSON, the string
    // "0" is different from the number 0.  If we conflate these, then later
    // expectations about annotation structure may be broken.  I.e., an
    // annotation expecting a string may see a number.
    if (!err && !unquotedValue.get().getAsNumber())
      return convertJSONToAttribute(context, unquotedValue.get(), p);
    // If there was an error, then swallow it and handle this as a string.
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {});
    return StringAttr::get(context, *a);
  }

  // Integer
  if (auto a = value.getAsInteger())
    return IntegerAttr::get(IntegerType::get(context, 64), *a);

  // Float
  if (auto a = value.getAsNumber())
    return FloatAttr::get(mlir::FloatType::getF64(context), *a);

  // Boolean
  if (auto a = value.getAsBoolean())
    return BoolAttr::get(context, *a);

  // Null
  if (auto a = value.getAsNull())
    return mlir::UnitAttr::get(context);

  // Object
  if (auto a = value.getAsObject()) {
    NamedAttrList metadata;
    for (auto b : *a)
      metadata.append(
          b.first, convertJSONToAttribute(context, b.second, p.field(b.first)));
    return DictionaryAttr::get(context, metadata);
  }

  // Array
  if (auto a = value.getAsArray()) {
    SmallVector<Attribute> metadata;
    for (size_t i = 0, e = (*a).size(); i != e; ++i)
      metadata.push_back(convertJSONToAttribute(context, (*a)[i], p.index(i)));
    return ArrayAttr::get(context, metadata);
  }

  llvm_unreachable("Impossible unhandled JSON type");
}

/// Convert a JSON value containing OMIR JSON (an array of OMNodes), convert
/// this to an OMIRAnnotation.
static Attribute convertJSONToOMIR(MLIRContext *context, json::Value &value,
                                   json::Path path) {
  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto *array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected OMIR to be an array of nodes, but found something else.");
    return {};
  }

  // Build a mutable map of Target to Annotation.
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto *object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected OMIR to be an array of objects, but found an array of "
               "something else.");
      return {};
    }

    // Manually built up OMNode.
    NamedAttrList omnode;

    // Validate that this looks like an OMNode.  This should have three fields:
    //   - "info": String
    //   - "id": String that starts with "OMID:"
    //   - "fields": Array<Object>
    // Fields is optional and is a dictionary encoded as an array of objects:
    //   - "info": String
    //   - "name": String
    //   - "value": JSON
    // The dictionary is keyed by the "name" member and the array of fields is
    // guaranteed to not have collisions of the "name" key.
    auto maybeInfo = object->getString("info");
    if (!maybeInfo) {
      p.report("OMNode missing mandatory member \"info\" with type \"string\"");
      return {};
    }
    auto maybeID = object->getString("id");
    if (!maybeID || !maybeID->startswith("OMID:")) {
      p.report("OMNode missing mandatory member \"id\" with type \"string\" "
               "that starts with \"OMID:\"");
      return {};
    }
    auto *maybeFields = object->get("fields");
    if (maybeFields && !maybeFields->getAsArray()) {
      p.report("OMNode has \"fields\" member with incorrect type (expected "
               "\"array\")");
      return {};
    }
    Attribute fields;
    if (!maybeFields)
      fields = DictionaryAttr::get(context, {});
    else {
      auto array = *maybeFields->getAsArray();
      NamedAttrList fieldAttrs;
      for (size_t i = 0, e = array.size(); i != e; ++i) {
        auto *field = array[i].getAsObject();
        auto pI = p.field("fields").index(i);
        if (!field) {
          pI.report("OMNode has field that is not an \"object\"");
          return {};
        }
        auto maybeInfo = field->getString("info");
        if (!maybeInfo) {
          pI.report(
              "OMField missing mandatory member \"info\" with type \"string\"");
          return {};
        }
        auto maybeName = field->getString("name");
        if (!maybeName) {
          pI.report(
              "OMField missing mandatory member \"name\" with type \"string\"");
          return {};
        }
        auto *maybeValue = field->get("value");
        if (!maybeValue) {
          pI.report("OMField missing mandatory member \"value\"");
          return {};
        }
        NamedAttrList values;
        values.append("info", StringAttr::get(context, *maybeInfo));
        values.append("value", convertJSONToAttribute(context, *maybeValue,
                                                      pI.field("value")));
        fieldAttrs.append(*maybeName, DictionaryAttr::get(context, values));
      }
      fields = DictionaryAttr::get(context, fieldAttrs);
    }

    omnode.append("info", StringAttr::get(context, *maybeInfo));
    omnode.append("id", convertJSONToAttribute(context, *object->get("id"),
                                               p.field("id")));
    omnode.append("fields", fields);
  }

  NamedAttrList omirAnnoFields;
  omirAnnoFields.append("class", StringAttr::get(context, omirAnnoClass));
  omirAnnoFields.append("nodes", convertJSONToAttribute(context, value, path));

  return DictionaryAttr::get(context, omirAnnoFields);
}

/// Get annotations or an empty set of annotations.
static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annotations = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annotations;
  return ArrayAttr::get(op->getContext(), {});
}

/// Construct the annotation array with a new thing appended.
static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

/// Update an ArrayAttribute by replacing one entry.
static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

/// Apply a new annotation to a resolved target.  This handles ports,
/// aggregates, modules, wires, etc.
static void addAnnotation(AnnoTarget ref, unsigned fieldIdx,
                          ArrayRef<NamedAttribute> anno) {
  auto *context = ref.getOp()->getContext();
  DictionaryAttr annotation;
  if (fieldIdx) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
        StringAttr::get(context, "circt.fieldID"),
        IntegerAttr::get(IntegerType::get(context, 32, IntegerType::Signless),
                         fieldIdx));
    annotation = DictionaryAttr::get(context, annoField);
  } else {
    annotation = DictionaryAttr::get(context, anno);
  }

  if (ref.isa<OpAnnoTarget>()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.getOp()), annotation);
    ref.getOp()->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portRef = ref.cast<PortAnnoTarget>();
  auto portAnnoRaw = ref.getOp()->getAttr(getPortAnnotationAttrName());
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.getOp())) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.getOp()),
        ArrayAttr::get(ref.getOp()->getContext(), {}));
    portAnno = ArrayAttr::get(ref.getOp()->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, portRef.getPortNo(),
      appendArrayAttr(portAnno[portRef.getPortNo()].dyn_cast<ArrayAttr>(),
                      annotation));
  ref.getOp()->setAttr("portAnnotations", portAnno);
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(const AnnoPathValue &target,
                                  ApplyState &state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> insts;
  for (auto inst : target.instances) {
    insts.push_back(OpAnnoTarget(inst).getNLAReference(
        state.getNamespace(inst->getParentOfType<FModuleLike>())));
  }

  insts.push_back(
      FlatSymbolRefAttr::get(target.ref.getModule().moduleNameAttr()));
  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);
  auto nla = b.create<HierPathOp>(state.circuit.getLoc(), "nla", instAttr);
  state.symTbl.insert(nla);
  return FlatSymbolRefAttr::get(nla);
}

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(const AnnoPathValue &target,
                                             ApplyState &state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);
  return sym;
}

//===----------------------------------------------------------------------===//
// Standard Utility Resolvers
//===----------------------------------------------------------------------===//

/// Always resolve to the circuit, ignoring the annotation.
static Optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                         ApplyState &state) {
  return AnnoPathValue(state.circuit);
}

/// Implementation of standard resolution.  First parses the target path, then
/// resolves it.
static Optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                              ApplyState &state) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    mlir::emitError(state.circuit.getLoc())
        << "Cannot tokenize annotation path " << rawPath;
    return {};
  }

  return resolveEntities(*tokens, state.circuit, state.symTbl,
                         state.targetCaches);
}

/// (SFC) FIRRTL SingleTargetAnnotation resolver.  Uses the 'target' field of
/// the annotation with standard parsing to resolve the path.  This requires
/// 'target' to exist and be normalized (per docs/FIRRTLAnnotations.md).
static Optional<AnnoPathValue> stdResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (!target) {
    mlir::emitError(state.circuit.getLoc())
        << "No target field in annotation " << anno;
    return {};
  }
  if (!target->getValue().isa<StringAttr>()) {
    mlir::emitError(state.circuit.getLoc())
        << "Target field in annotation doesn't contain string " << anno;
    return {};
  }
  return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                        state);
}

/// Resolves with target, if it exists.  If not, resolves to the circuit.
static Optional<AnnoPathValue> tryResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                          state);
  return AnnoPathValue(state.circuit);
}

//===----------------------------------------------------------------------===//
// Standard Utility Appliers
//===----------------------------------------------------------------------===//

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotation.  Optionally handles non-local annotations.
static LogicalResult applyWithoutTargetImpl(const AnnoPathValue &target,
                                            DictionaryAttr anno,
                                            ApplyState &state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal()) {
    Annotation annotation(anno);
    auto diag = mlir::emitError(target.ref.getOp()->getLoc())
                << "is targeted by a non-local annotation \""
                << annotation.getClass() << "\" with target "
                << annotation.getMember("target")
                << ", but this annotation cannot be non-local";
    diag.attachNote() << "see current annotation: " << anno << "\n";
    return failure();
  }
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.getName().getValue() != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {StringAttr::get(anno.getContext(), "circt.nonlocal"), sym});
    }
  }
  addAnnotation(target.ref, target.fieldIdx, newAnnoAttrs);
  return success();
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotation.  Optionally handles non-local annotations.
/// Ensures the target resolves to an expected type of operation.
template <bool allowNonLocal, bool allowPortAnnoTarget, typename T,
          typename... Tr>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  if (target.ref.isa<PortAnnoTarget>()) {
    if (!allowPortAnnoTarget)
      return failure();
  } else if (!target.isOpOfType<T, Tr...>())
    return failure();

  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTarget<allowNonLocal, false, T, Tr...>(target, anno,
                                                            state);
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(const AnnoPathValue &target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

/// Just drop the annotation.  This is intended for Annotations which are known,
/// but can be safely ignored.
static LogicalResult drop(const AnnoPathValue &target, DictionaryAttr anno,
                          ApplyState &state) {
  return success();
}

/// Resolution and application of a "firrtl.annotations.NoTargetAnnotation".
/// This should be used for any Annotation which does not apply to anything in
/// the FIRRTL Circuit, i.e., an Annotation which has no target.  Historically,
/// NoTargetAnnotations were used to control the Scala FIRRTL Compiler (SFC) or
/// its passes, e.g., to set the output directory or to turn on a pass.
/// Examples of these in the SFC are "firrtl.options.TargetDirAnnotation" to set
/// the output directory or "firrtl.stage.RunFIRRTLTransformAnnotation" to
/// casuse the SFC to schedule a specified pass.  Instead of leaving these
/// floating or attaching them to the top-level MLIR module (which is a purer
/// interpretation of "no target"), we choose to attach them to the Circuit even
/// they do not "apply" to the Circuit.  This gives later passes a common place,
/// the Circuit, to search for these control Annotations.
static AnnoRecord NoTargetAnnotation = {noResolve,
                                        applyWithoutTarget<false, CircuitOp>};

//===----------------------------------------------------------------------===//
// Driving table
//===----------------------------------------------------------------------===//

static llvm::StringMap<AnnoRecord> createDrivingTable() {
  return {
      // Testing Annotation
      {"circt.test", {stdResolve, applyWithoutTarget<true>}},
      {"circt.testLocalOnly", {stdResolve, applyWithoutTarget<>}},
      {"circt.testNT", {noResolve, applyWithoutTarget<>}},
      {"circt.missing", {tryResolve, applyWithoutTarget<true>}},
      // Grand Central Views/Interfaces Annotations
      {extractGrandCentralClass, NoTargetAnnotation},
      {grandCentralHierarchyFileAnnoClass, NoTargetAnnotation},
      {serializedViewAnnoClass, {noResolve, applyGCTView}},
      {viewAnnoClass, {noResolve, applyGCTView}},
      {companionAnnoClass, {stdResolve, applyWithoutTarget<>}},
      {parentAnnoClass, {stdResolve, applyWithoutTarget<>}},
      {augmentedGroundTypeClass, {stdResolve, applyWithoutTarget<true>}},
      // Grand Central Data Tap Annotations
      {dataTapsClass, {noResolve, applyGCTDataTaps}},
      {dataTapsBlackboxClass, {stdResolve, applyWithoutTarget<true>}},
      {referenceKeySourceClass, {stdResolve, applyWithoutTarget<true>}},
      {referenceKeyPortClass, {stdResolve, applyWithoutTarget<true>}},
      {internalKeySourceClass, {stdResolve, applyWithoutTarget<true>}},
      {internalKeyPortClass, {stdResolve, applyWithoutTarget<true>}},
      {deletedKeyClass, {stdResolve, applyWithoutTarget<true>}},
      {literalKeyClass, {stdResolve, applyWithoutTarget<true>}},
      // Grand Central Mem Tap Annotations
      {memTapClass, {noResolve, applyGCTMemTaps}},
      {memTapSourceClass, {stdResolve, applyWithoutTarget<true>}},
      {memTapPortClass, {stdResolve, applyWithoutTarget<true>}},
      {memTapBlackboxClass, {stdResolve, applyWithoutTarget<true>}},
      // Grand Central Signal Mapping Annotations
      {signalDriverAnnoClass, {noResolve, applyGCTSignalMappings}},
      {signalDriverTargetAnnoClass, {stdResolve, applyWithoutTarget<true>}},
      {signalDriverModuleAnnoClass, {stdResolve, applyWithoutTarget<true>}},
      // OMIR Annotations
      {omirAnnoClass, {noResolve, applyOMIR}},
      {omirTrackerAnnoClass, {stdResolve, applyWithoutTarget<true>}},
      {omirFileAnnoClass, NoTargetAnnotation},
      // Miscellaneous Annotations
      {dontTouchAnnoClass,
       {stdResolve, applyWithoutTarget<true, true, WireOp, NodeOp, RegOp,
                                       RegResetOp, InstanceOp, MemOp, CombMemOp,
                                       MemoryPortOp, SeqMemOp>}},
      {prefixModulesAnnoClass,
       {stdResolve,
        applyWithoutTarget<true, FModuleOp, FExtModuleOp, InstanceOp>}},
      {dutAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
      {extractSeqMemsAnnoClass, NoTargetAnnotation},
      {injectDUTHierarchyAnnoClass, NoTargetAnnotation},
      {convertMemToRegOfVecAnnoClass, NoTargetAnnotation},
      {excludeMemToRegAnnoClass,
       {stdResolve, applyWithoutTarget<true, MemOp, CombMemOp>}},
      {sitestBlackBoxAnnoClass, NoTargetAnnotation},
      {enumComponentAnnoClass, {noResolve, drop}},
      {enumDefAnnoClass, {noResolve, drop}},
      {enumVecAnnoClass, {noResolve, drop}},
      {forceNameAnnoClass,
       {stdResolve, applyWithoutTarget<true, FModuleOp, FExtModuleOp>}},
      {flattenAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
      {inlineAnnoClass, {stdResolve, applyWithoutTarget<false, FModuleOp>}},
      {noDedupAnnoClass,
       {stdResolve, applyWithoutTarget<false, FModuleOp, FExtModuleOp>}},
      {blackBoxInlineAnnoClass,
       {stdResolve, applyWithoutTarget<false, FExtModuleOp>}},
      {dontObfuscateModuleAnnoClass,
       {stdResolve, applyWithoutTarget<false, FModuleOp>}},
      {verifBlackBoxAnnoClass,
       {stdResolve, applyWithoutTarget<false, FExtModuleOp>}},
      {elaborationArtefactsDirectoryAnnoClass, NoTargetAnnotation},
      {subCircuitsTargetDirectoryAnnoClass, NoTargetAnnotation},
      {retimeModulesFileAnnoClass, NoTargetAnnotation},
      {retimeModuleAnnoClass,
       {stdResolve, applyWithoutTarget<false, FModuleOp, FExtModuleOp>}},
      {metadataDirectoryAttrName, NoTargetAnnotation},
      {moduleHierAnnoClass, NoTargetAnnotation},
      {sitestTestHarnessBlackBoxAnnoClass, NoTargetAnnotation},
      {testBenchDirAnnoClass, NoTargetAnnotation},
      {testHarnessHierAnnoClass, NoTargetAnnotation},
      {testHarnessPathAnnoClass, NoTargetAnnotation},
      {prefixInterfacesAnnoClass, NoTargetAnnotation},
      {subCircuitDirAnnotation, NoTargetAnnotation},
      {extractAssertAnnoClass, NoTargetAnnotation},
      {extractAssumeAnnoClass, NoTargetAnnotation},
      {extractCoverageAnnoClass, NoTargetAnnotation},
      {dftTestModeEnableAnnoClass, {stdResolve, applyWithoutTarget<true>}},
      {runFIRRTLTransformAnnoClass, {noResolve, drop}},
      {mustDedupAnnoClass, NoTargetAnnotation},
      {addSeqMemPortAnnoClass, NoTargetAnnotation},
      {addSeqMemPortsFileAnnoClass, NoTargetAnnotation},
      {extractClockGatesAnnoClass, NoTargetAnnotation},
      {fullAsyncResetAnnoClass, {stdResolve, applyWithoutTarget<true>}},
      {ignoreFullAsyncResetAnnoClass,
       {stdResolve, applyWithoutTarget<true, FModuleOp>}},
      {decodeTableAnnotation, {noResolve, drop}},
      {blackBoxTargetDirAnnoClass, NoTargetAnnotation}};
}

//===----------------------------------------------------------------------===//
// ApplyState
//===----------------------------------------------------------------------===//

ApplyState::ApplyState(const FIRParserOptions &options, CircuitOp circuit,
                       AnnotationParser *parser)
    : context(circuit.getContext()), circuit(circuit), symTbl(circuit),
      instanceGraph(circuit), instancePathCache(instanceGraph), parser(parser) {
}

//===----------------------------------------------------------------------===//
// AnnotationParser
//===----------------------------------------------------------------------===//

AnnotationParser::AnnotationParser(const FIRParserOptions &options,
                                   CircuitOp circuit)
    : context(circuit.getContext()), circuit(circuit), options(options),
      circuitTarget(("~" + circuit.getName()).str()),
      applyState(options, circuit, this),
      annotationRecords(createDrivingTable()) {}

/// Lookup a record for a given annotation class.  Optionally, returns the
/// record for "circuit.missing" if the record doesn't exist.
const AnnoRecord *
AnnotationParser::getAnnotationHandler(StringRef annoStr) const {
  auto ii = annotationRecords.find(annoStr);
  if (ii != annotationRecords.end())
    return &ii->second;
  return nullptr;
}

LogicalResult AnnotationParser::applyAnnotation(DictionaryAttr anno) {
  // Lookup the class
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->getValue().cast<StringAttr>().getValue();
  else if (options.ignoreClasslessAnnotations)
    annoClassVal = "circt.missing";
  else
    return mlir::emitError(circuit.getLoc())
           << "Annotation without a class: " << anno;

  // See if we handle the class.
  auto *record = getAnnotationHandler(annoClassVal);
  if (!record && !options.ignoreUnhandledAnnotations) {
    return mlir::emitError(circuit.getLoc())
           << "Unhandled annotation: " << anno;

    // Try again, requesting the fallback handler.
    record = getAnnotationHandler("circt.missing");
    assert(record);
  }

  // Try to apply the annotation.
  auto target = record->resolver(anno, applyState);
  if (!target)
    return mlir::emitError(circuit.getLoc())
           << "Unable to resolve target of annotation: " << anno;
  if (record->applier(*target, anno, applyState).failed())
    return mlir::emitError(circuit.getLoc())
           << "Unable to apply annotation: " << anno;
  return success();
}

LogicalResult AnnotationParser::parseAnnotations(Location loc,
                                                 StringRef annotationsStr) {
  // Parse the annotations into JSON.
  auto json = json::parse(annotationsStr);
  if (auto err = json.takeError()) {
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {
      auto diag = emitError(loc, "Failed to parse JSON Annotations");
      diag.attachNote() << a.message();
    });
    return failure();
  }

  json::Path::Root root;
  json::Path path = root;

  auto *annotations = json->getAsArray();

  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  if (!annotations) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    auto diag = emitError(loc, "Invalid/unsupported annotation format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(json.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }

  // Process each annotation.
  for (size_t i = 0, e = annotations->size(); i != e; ++i) {
    auto p = path.index(i);
    auto attr = convertJSONToAttribute(context, (*annotations)[i], path);
    llvm::errs() << "processing: " << attr << "\n";

    // Make sure that this was a JSON object.
    auto anno = attr.dyn_cast<DictionaryAttr>();
    if (!anno) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      auto diag = emitError(loc, "Invalid/unsupported annotation format");
      std::string jsonErrorMessage =
          "See inline comments for problem area in JSON:\n";
      llvm::raw_string_ostream s(jsonErrorMessage);
      root.printErrorContext(json.get(), s);
      diag.attachNote() << jsonErrorMessage;
      return failure();
    }

    // Process the annotation.
    if (failed(applyAnnotation(anno)))
      return failure();

    // Attach any additional annotation targets to the circuit.
    while (!worklist.empty()) {
      if (failed(applyAnnotation(worklist.pop_back_val())))
        return failure();
    }
  }
  return success();
}

LogicalResult AnnotationParser::parseOMIR(Location loc, StringRef omirStr) {
  // Parse the annotations into JSON.
  auto json = json::parse(omirStr);
  if (auto err = json.takeError()) {
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {
      auto diag = emitError(loc, "Failed to parse JSON Annotations");
      diag.attachNote() << a.message();
    });
    return failure();
  }

  json::Path::Root root;
  json::Path path = root;

  auto attr = convertJSONToOMIR(context, json.get(), path);

  // Check if the OMIR JSON failed to deserialize.
  if (!attr) {
    auto diag = emitError(loc, "Invalid/unsupported annotation format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(json.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }
  llvm::errs() << "omir: " << attr << "\n";

  // Make sure that this was a JSON object.
  auto anno = attr.dyn_cast<DictionaryAttr>();
  if (!anno) {
    path.report("Expected annotations to be an array of objects, but found an "
             "array of something else.");
    auto diag = emitError(loc, "Invalid/unsupported annotation format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(json.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }

  // Process the annotation.
  if (failed(applyAnnotation(anno)))
    return failure();

  // Attach any additional annotation targets to the circuit.
  while (!worklist.empty()) {
    if (failed(applyAnnotation(worklist.pop_back_val())))
      return failure();
  }
  return success();
}