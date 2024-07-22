#include "circt/HWL/Incremental/Database.h"
#include "gtest/gtest.h"
#include <unordered_map>

using namespace circt;
using namespace hwl;

namespace example {

struct Database;
// using ContextBase = inc::ContextBase<Database>;
using Context = inc::Context<Database>;
using QueryContext = inc::QueryContext<Database>;

struct FileContents {
  FileContents() = default;
  explicit FileContents(std::string data) : data(std::move(data)) {}
  std::string data = "";
};

struct AST {
  bool operator==(const AST &rhs) const { return true; }
};

struct GetAST {
  AST operator()(QueryContext ctx, const std::string &filename) const;
};

struct TypeCheck {
  struct Result {
    bool operator==(Result rhs) const { return ok == rhs.ok; }
    bool ok;
  };
  Result operator()(QueryContext ctx, const std::string &filename) const;
};

struct Database : public inc::Database {
  inc::Input<FileContents> fileContents = "";
  inc::Query<GetAST> getAST;
  inc::Query<TypeCheck> typeCheck;
};

inline AST GetAST::operator()(QueryContext ctx,
                              const std::string &filename) const {
  //   ctx->fileContentsTable.get(ctx, filename).get()->get(ctx);
  llvm::errs() << "getting AST\n";
  ctx->fileContents.get(ctx);
  return {};
}

inline auto TypeCheck::operator()(QueryContext ctx,
                                  const std::string &filename) const -> Result {
  auto ast = ctx->getAST(ctx, filename);
  (void)ast;
  return Result{true};
}

} // namespace example

TEST(Incremental, Basic) {
  example::Database db;
  example::Context ctx(&db);
  auto ast = ctx->typeCheck(ctx, std::string("foo.test"));
  ast = ctx->typeCheck(ctx, std::string("foo.test"));
  ctx->fileContents.modify(ctx) = example::FileContents("hello");
  ast = ctx->typeCheck(ctx, std::string("foo.test"));

  (void)ast;
}
