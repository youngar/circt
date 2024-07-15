#include "circt/HWML/Incremental/Database.h"
#include "gtest/gtest.h"
#include <unordered_map>

using namespace circt;
using namespace hwml;

namespace example {

struct Database;
// using ContextBase = inc::ContextBase<Database>;
using Context = inc::Context<Database>;
using QueryContext = inc::QueryContext<Database>;

struct FileContents {
  std::string data;
};

class FileContentsTable {
  using Table = std::unordered_map<std::string,
                                   std::unique_ptr<inc::Input<FileContents>>>;

public:
  //   template <typename D>
  //   const FileContents &get(ContextBase ctx, const std::string &filename)
  //   const {
  //     auto &cached = table[filename];
  //     if (cached)
  //       return cached->get(ctx);

  //     std::unique_ptr entry = new inc::Input<FileContents>(ctx, "contents");
  //   }

  mutable Table table;
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

struct Database : inc::Database {
  //   FileContentsTable fileContentsTable;
  inc::Query<GetAST> getAST;
  inc::Query<TypeCheck> typeCheck;
};

inline AST GetAST::operator()(QueryContext ctx,
                              const std::string &filename) const {
  //   ctx->fileContentsTable.get(ctx, filename).get()->get(ctx);
  llvm::errs() << "getting AST\n";
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
  (void)ast;
}
