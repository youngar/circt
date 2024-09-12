#ifndef CIRCT_HWL_HWLPARSER_H
#define CIRCT_HWL_HWLPARSER_H

#include "circt/HWL/HWLDiagnostic.h"
#include "circt/HWL/HWLLocation.h"
#include "circt/HWL/Incremental/Database.h"
#include "circt/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hwl {

///
/// Concrete Syntax Tree
///

namespace cst {

enum class Associativity { left, right, none };

enum class Fixity { prefix, infix, postfix, closed };

struct Operator {
  /// Get the number of internal arguments.  This does not count leading or
  /// trailing arguments.
  std::size_t getInteralArity() { return nameparts.size() - 1; }
  Associativity associativity;
  Fixity fixity;
  std::vector<std::string> nameparts;
};

struct PrecedenceNode {
  std::vector<Operator> operators;
  std::vector<PrecedenceNode *> successors;
};

struct PrecedenceGraph {
  // llvm::BumpPtrList<PrecedenceNode> nodes;
};

struct Expr {};

struct Binder {
  std::string name;
  Expr type;
};

struct Notation {};

struct TypeConsDecl {
  std::string name;
};

struct TypeDecl {
  std::string name;
};

///
/// Statements
///

struct Statement {
};

struct DefArg {};

struct Def {
  std::string name;
  std::vector<DefArg> args;
  Expr type;
  Expr body;
};

///
/// File
///

enum class NodeKind {
  Type,
  Def,
};

struct Tag {
  enum class Kind {
    Def,
  };
  Statement() = delete;
  Statement(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

privarte:
  Kind kind
};

struct Mixin {

};

struct Def : Node {

};
struct File {

  def
  private:
  llvm::BumpPtrAllocator allocation;
};

} // namespace cst

struct HWLParser;
struct HWLDocument;

struct HWLParseResult {
  bool success;
  cst::File tree;
  std::vector<Diagnostic> diagnostics;
};

struct HWLParser {

  HWLParser();

  HWLParseResult parse(const HWLDocument &document);

private:
};

struct HWLDocument {
  HWLDocument(const std::string &contents) : contents(contents) {}
  HWLDocument(const HWLDocument &) = delete;
  HWLDocument &operator=(const HWLDocument &) = delete;

  std::string getContents() const { return contents; }
  void replaceContents(const std::string &contents);

private:
  friend HWLParser;

  /// The full string contents of the file.
  std::string contents;
};

} // namespace hwl
} // namespace circt
#endif
