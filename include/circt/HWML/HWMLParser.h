#ifndef CIRCT_HWML_HWMLPARSER_H
#define CIRCT_HWML_HWMLPARSER_H

#include "circt/HWML/Parse/Machine.h"
#include "circt/HWML/Parse/MemoTable.h"
#include "circt/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace hwml {

template <typename Derived, typename Parent>
class Kind : public Parent {

protected:
  Kind() : Parent(getQueryID()) {}

  static TypeID getQueryID() { return TypeID::get<Derived>(); }

  static bool classof(const Parent *node) {
    return node->getTypeID() == getQueryID();
  }
};

class Expression {
public:
  TypeID getTypeID() const { return typeID; }

protected:
  Expression(TypeID typeID) : typeID(typeID) {}

private:
  TypeID typeID;
};

class Identifier : public Kind<Identifier, Expression> {
public:
  Identifier(StringRef id) : id(id) {}
  StringRef get() { return id; }

private:
  StringRef id;
};

class Application : public Kind<Application, Expression> {
public:
  Application(Expression *head, Expression *operand)
      : head(head), operand(operand) {}
  Expression *getHead() { return head; }
  Expression *getOperand() { return operand; }

private:
  Expression *head;
  Expression *operand;
};

class Statement {
public:
  TypeID getTypeID() const { return typeID; }

protected:
  Statement(TypeID typeID) : typeID(typeID) {}

private:
  TypeID typeID;
};

class Declaration : Kind<Declaration, Statement> {
public:
  Declaration(StringRef name, Expression *type) : name(name), type(type) {}
  StringRef getName() { return name; }
  Expression *getType() { return type; }

private:
  StringRef name;
  Expression *type;
};

class Definition : Kind<Definition, Statement> {
  public:

  private:
};

class CompUnit {
  std::vector<Statement> statements;
};

struct HWMLParser {
  HWMLParser();

  bool parse(StringRef contents, std::vector<Capture> &captures,
             std::vector<Diagnostic> &diagnostics);

private:
  MemoTable memoTable;
};

} // namespace hwml
} // namespace circt
#endif
