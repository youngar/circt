#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/HWMLIncremental.h"
#include "llvm/ADT/StringMap.h"
#include <iostream>

using namespace circt;
using namespace hwml;

///
/// Helper
///

template <typename ConcreteType>
class KindMixin {
  static bool classof(const KindMixin *base) {
    return base->typeID == TypeID::get<ConcreteType>();
  }
};

///
/// CST
///

class CSTNode {
  TypeID typeID;
};

class CSTFile : public CSTNode, public KindMixin<CSTFile> {};

///
/// CST
///

class ConcreteSyntax {};

class File {
public:
  explicit File(std::string contents) : contents(contents) {}

private:
  std::string contents;
};

class Ast;
class ConcreteSyntaxDB {
public:
private:
  llvm::StringMap<Ast> files;
};

///
/// Ast
///

class Ast {
  Expr *expr;
  bool valid = false;
};

class AstDB {
public:
  File openFile(std::string filename, std::string contents) {}

  Ast query(const File &file) const {}

  void update(File &file) const {}

private:
};

///
/// Eval
///

class EvalParams {
  Ast ast;
};

/// Stores the canonical forms of expressions?
class EvalCacheEntry {
  Ast result;
  // Hash inputsHash;
};

class EvalDB {
  DenseMap<EvalParams, EvalCacheEntry> evals;
};

int main(int argc, char *argv[]) {
  const char *program = "hwl";
  const char *filename = nullptr;
  switch (argc) {
  case 0:
    break;
  case 1:
    program = argv[0];
    break;
  case 2:
    program = argv[0];
    filename = argv[1];
    break;
  default:
    break;
  }

  if (!filename) {
    // llvm::errs() << program << ": missing <filename>\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}