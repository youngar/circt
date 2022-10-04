//===- FIRRTLAnnotationHelper.h - FIRRTL Annotation Lookup ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers mapping annotations to operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H

#include "FIRParser.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinAttributes.h>

namespace circt {
namespace firrtl {

/// Stores an index into an aggregate.
struct TargetToken {
  StringRef name;
  bool isIndex;
};

/// The parsed annotation path.
struct TokenAnnoTarget {
  StringRef circuit;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef module;
  // The final name of the target
  StringRef name;
  // Any aggregates indexed.
  SmallVector<TargetToken> component;

  /// Append the annotation path to the given `SmallString` or `SmallVector`.
  void toVector(SmallVectorImpl<char> &out) const;

  /// Convert the annotation path to a string.
  std::string str() const {
    SmallString<32> out;
    toVector(out);
    return std::string(out);
  }
};

// The potentially non-local resolved annotation.
struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  AnnoTarget ref;
  unsigned fieldIdx = 0;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(Operation *op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, AnnoTarget b,
                unsigned fieldIdx)
      : instances(insts.begin(), insts.end()), ref(b), fieldIdx(fieldIdx) {}

  bool isLocal() const { return instances.empty(); }

  template <typename... T>
  bool isOpOfType() const {
    if (auto opRef = ref.dyn_cast<OpAnnoTarget>())
      return isa<T...>(opRef.getOp());
    return false;
  }
};

/// Cache AnnoTargets for a module's named things.
struct AnnoTargetCache {
  AnnoTargetCache() = delete;
  AnnoTargetCache(const AnnoTargetCache &other) = default;
  AnnoTargetCache(AnnoTargetCache &&other)
      : targets(std::move(other.targets)){};

  AnnoTargetCache(FModuleLike mod) { gatherTargets(mod); };

  /// Lookup the target for 'name', empty if not found.
  /// (check for validity using operator bool()).
  AnnoTarget getTargetForName(StringRef name) const {
    return targets.lookup(name);
  }

  void insertOp(Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp,
              chirrtl::CombMemOp, chirrtl::SeqMemOp, chirrtl::MemoryPortOp,
              chirrtl::MemoryDebugPortOp, PrintFOp>([&](auto op) {
          // To be safe, check attribute and non-empty name before adding.
          if (auto name = op.getNameAttr(); name && !name.getValue().empty())
            targets.insert({name, OpAnnoTarget(op)});
        });
  }

  /// Replace `oldOp` with `newOp` in the target cache. The new and old ops can
  /// have different names.
  void replaceOp(Operation *oldOp, Operation *newOp) {
    if (auto name = oldOp->getAttrOfType<StringAttr>("name");
        name && !name.getValue().empty())
      targets.erase(name);
    insertOp(newOp);
  }

  /// Add a new module port to the target cache.
  void insertPort(FModuleLike mod, size_t portNo) {
    targets.insert({mod.getPortNameAttr(portNo), PortAnnoTarget(mod, portNo)});
  }

private:
  /// Walk the module and add named things to 'targets'.
  void gatherTargets(FModuleLike mod);

  llvm::DenseMap<StringRef, AnnoTarget> targets;
};

/// Cache AnnoTargets for a circuit's modules, walked as needed.
struct CircuitTargetCache {
  /// Get cache for specified module, creating it as needed.
  /// Returned reference may become invalidated by future calls.
  const AnnoTargetCache &getOrCreateCacheFor(FModuleLike module) {
    auto it = targetCaches.find(module);
    if (it == targetCaches.end())
      it = targetCaches.try_emplace(module, module).first;
    return it->second;
  }

  /// Lookup the target for 'name' in 'module'.
  AnnoTarget lookup(FModuleLike module, StringRef name) {
    return getOrCreateCacheFor(module).getTargetForName(name);
  }

  /// Clear the cache completely.
  void invalidate() { targetCaches.clear(); }

  /// Replace `oldOp` with `newOp` in the target cache. The new and old ops can
  /// have different names.
  void replaceOp(Operation *oldOp, Operation *newOp) {
    auto mod = newOp->getParentOfType<FModuleOp>();
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().replaceOp(oldOp, newOp);
  }

  /// Add a new module port to the target cache.
  void insertPort(FModuleLike mod, size_t portNo) {
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().insertPort(mod, portNo);
  }

  /// Add a new op to the target cache.
  void insertOp(Operation *op) {
    auto mod = op->getParentOfType<FModuleOp>();
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().insertOp(op);
  }

private:
  DenseMap<Operation *, AnnoTargetCache> targetCaches;
};

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
std::string canonicalizeTarget(StringRef target);

/// Parse a FIRRTL annotation path into its constituent parts.
Optional<TokenAnnoTarget> tokenizePath(StringRef origTarget);

/// Convert a parsed target string to a resolved target structure.  This
/// resolves all names and aggregates from a parsed target.
Optional<AnnoPathValue> resolveEntities(TokenAnnoTarget path, CircuitOp circuit,
                                        SymbolTable &symTbl,
                                        CircuitTargetCache &cache);

/// Resolve a string path to a named item inside a circuit.
Optional<AnnoPathValue> resolvePath(StringRef rawPath, CircuitOp circuit,
                                    SymbolTable &symTbl,
                                    CircuitTargetCache &cache);

class AnnotationParser;

/// State threaded through functions for resolving and applying annotations.
class ApplyState {
public:
  ApplyState(const FIRParserOptions &options, CircuitOp circuit,
             AnnotationParser *parser);

  MLIRContext *context;
  CircuitOp circuit;
  SymbolTable symTbl;
  CircuitTargetCache targetCaches;
  InstanceGraph instanceGraph;
  InstancePathCache instancePathCache;
  DenseMap<Attribute, FlatSymbolRefAttr> instPathToNLAMap;
  size_t numReusedHierPaths = 0;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

  /// Get the next unique id and advance to the next ID.  Unique IDs can be used
  /// to tie different annotations together.
  IntegerAttr newID() {
    return IntegerAttr::get(IntegerType::get(circuit.getContext(), 64),
                            annotationID++);
  };

  /// Takes an annotation an applies it's registered handler to attach it to
  /// the IR.
  void addToWorklist(DictionaryAttr anno);

private:
  /// A pointer to the parser which owns this.  Used to add more annotations
  /// to the worklist.
  AnnotationParser *parser;

  /// Maps a module to its namespace.
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;

  /// The next available unique identifier for use by annotations.
  unsigned annotationID = 0;
};

///
struct AnnoRecord {
  ///
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, ApplyState &)>
      resolver;
  ///
  llvm::function_ref<LogicalResult(const AnnoPathValue &, DictionaryAttr,
                                   ApplyState &)>
      applier;
};

/// A parser which can import annotations and attach them to the circuit.
class AnnotationParser {
public:
  AnnotationParser(const FIRParserOptions &options, CircuitOp circuit);

  /// Parse a JSON encoded array of annotations.
  LogicalResult parseAnnotations(Location loc, StringRef annotationsStr);

  /// Parse a JSON encoded array of OMIR.
  LogicalResult parseOMIR(Location loc, StringRef omirStr);

private:
  // ApplyState needs to be able to queue more annotations on to the worklist.
  friend class ApplyState;

  /// Takes an annotation an applies it's registered handler to attach it to
  /// the IR.
  void addToWorklist(DictionaryAttr anno) { worklist.push_back(anno); }

  /// Get the annotation handler associated with an annotation's class.
  const AnnoRecord *getAnnotationHandler(StringRef annoStr) const;

  /// This will call the annotation handler associated with the annotation.
  /// Typically this will attach the annotation to the target.
  LogicalResult applyAnnotation(DictionaryAttr anno);

  MLIRContext *context;

  /// The circuit which we are parsing annotations for. All annotations targets
  /// should reference this circuit.
  CircuitOp circuit;

  /// The parsing options.
  const FIRParserOptions &options;

  /// A target representing the current circuit we are parsing.
  std::string circuitTarget;

  /// This is the apply state used for each annotation parser.
  ApplyState applyState;

  /// A list ok annotations which still need to be handled.  Annotations are
  /// allowed to add new annotations to the worklist as they are handled.
  SmallVector<DictionaryAttr> worklist;

  /// A mapping from annotation class strings to their handlers.
  llvm::StringMap<AnnoRecord> annotationRecords;
};

inline void ApplyState::addToWorklist(DictionaryAttr anno) {
  parser->addToWorklist(anno);
}

LogicalResult applyGCTView(const AnnoPathValue &target, DictionaryAttr anno,
                           ApplyState &state);

LogicalResult applyGCTDataTaps(const AnnoPathValue &target, DictionaryAttr anno,
                               ApplyState &state);

LogicalResult applyGCTMemTaps(const AnnoPathValue &target, DictionaryAttr anno,
                              ApplyState &state);

LogicalResult applyGCTSignalMappings(const AnnoPathValue &target,
                                     DictionaryAttr anno, ApplyState &state);

LogicalResult applyOMIR(const AnnoPathValue &target, DictionaryAttr anno,
                        ApplyState &state);

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the
/// value of a specific type associated with a key in a dictionary. However,
/// this is specialized to print a useful error message, specific to custom
/// annotation process, on failure.
template <typename A>
A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
           Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = value.dyn_cast_or_null<A>();
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

/// Add ports to the module and all its instances and return the clone for
/// `instOnPath`. This does not connect the new ports to anything. Replace
/// the old instances with the new cloned instance in all the caches.
InstanceOp addPortsToModule(
    FModuleOp mod, InstanceOp instOnPath, FIRRTLType portType, Direction dir,
    StringRef newName, InstancePathCache &instancePathcache,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace,
    CircuitTargetCache *targetCaches = nullptr);

/// Add a port to each instance on the path `instancePath` and forward the
/// `fromVal` through them. It returns the port added to the last module on the
/// given path. The module referenced by the first instance on the path must
/// contain `fromVal`.
Value borePortsOnPath(
    SmallVector<InstanceOp> &instancePath, FModuleOp lcaModule, Value fromVal,
    StringRef newNameHint, InstancePathCache &instancePathcache,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace,
    CircuitTargetCache *targetCachesInstancePathCache);

/// Find the lowest-common-ancestor `lcaModule`, between `srcTarget` and
/// `dstTarget`, and set `pathFromSrcToWire` with the path between them through
/// the `lcaModule`. The assumption here is that the srcTarget and dstTarget can
/// be uniquely identified. Either the instnaces field of their AnnoPathValue is
/// set or there exists a single path from Top.
LogicalResult findLCAandSetPath(AnnoPathValue &srcTarget,
                                AnnoPathValue &dstTarget,
                                SmallVector<InstanceOp> &pathFromSrcToWire,
                                FModuleOp &lcaModule, ApplyState &state);
} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
