#ifndef CIRCT_HWL_HWLTYPECHECK_H
#define CIRCT_HWL_HWLTYPECHECK_H

#include "circt/HWL/HWLAst.h"
#include "circt/HWL/HWLDatabase.h"
#include "circt/HWL/HWLParser.h"

namespace circt {
namespace hwl {

Expr *eval(Expr *expr);

///
/// Type Checking
///

Type *typeSynth(std::vector<Type *> ctxt, Expr *expr);

bool typeCheck(std::vector<Type *> ctxt, Expr *expr, Type *type);

struct GetASTFromCST {
  CompUnit operator()(QueryContext ctx, HWLParseResult cst);
};

struct TypeCheckExpr {
  std::vector<Diagnostic> operator()(QueryContext ctx, const Expr *expr);
};

struct TypeCheckStmt {
  std::vector<Diagnostic> operator()(QueryContext ctx, const Stmt *stmt);
};

struct TypeCheckCompUnit {
  std::vector<Diagnostic> operator()(QueryContext ctx,
                                     const CompUnit *compUnit);
};

struct GetDiagnosticsForFile {
  std::vector<Diagnostic> operator()(QueryContext ctx,
                                     const std::string *filename);
};

std::vector<Diagnostic> getDiagnostics(const HWLDatabase &database);

} // namespace hwl
} // namespace circt

#endif