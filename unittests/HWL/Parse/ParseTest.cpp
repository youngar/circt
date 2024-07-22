#include "circt/HWL/HWLAst.h"
#include "circt/HWL/Parse/Insn.h"
#include "circt/HWL/Parse/Machine.h"
#include "gtest/gtest.h"

using namespace circt::hwl;
using namespace circt::hwl::p;

#if 0
TEST(Machine, Match) {
  InsnStream stream;
  stream.match('a');
  stream.end();
  auto program = stream.finalize();

  const uint8_t sp[] = "a";
  auto *se = sp + sizeof(sp);
  std::vector<Node *> captures;
  std::vector<Node *> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  EXPECT_TRUE(result);
}

TEST(Machine, Choice) {
  InsnStream stream;
  auto l0 = stream.label();
  auto l1 = stream.label();
  stream.choice(l0);
  stream.match('a');
  stream.match('b');
  stream.match('c');
  stream.commit(l1);
  stream.setLabel(l0, stream.match('x'));
  stream.match('y');
  stream.match('z');
  stream.setLabel(l1, stream.end());

  auto program = stream.finalize();
  print(std::cerr, program);
  const uint8_t sp[] = "abc";
  auto *se = sp + sizeof(sp);
  std::cerr << "sp=" << (void *)sp << ", se=" << (void *)se << "\n";
  std::vector<Node *> captures;
  std::vector<Node *> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  EXPECT_TRUE(result);
}

TEST(Inc, Basic) {
  auto *ast = new App(new Abs(new Var(0)), new Const("A"));
  if (ast)
    std::cout << "hello\n";
  print(ast);
  print(eval(ast));
}

TEST(Machine, Capture) {
  InsnStream stream;

  auto exp = stream.label();
  auto factor = stream.label();
  auto term = stream.label();
  auto number = stream.label();

  // Top level
  stream.call(exp);
  stream.end();

  // Expr
  auto e0 = stream.label();
  auto e1 = stream.label();
  stream.setLabel(exp, stream.captureBegin(10));
  stream.call(factor);
  stream.choice(e1);
  stream.setLabel(e0, stream.set("+-"));
  stream.call(factor);
  stream.partialCommit(e0);
  stream.setLabel(e1, stream.captureEnd());
  stream.ret();

  // Factor
  auto f0 = stream.label();
  auto f1 = stream.label();
  stream.setLabel(factor, stream.captureBegin(11));
  stream.call(term);
  stream.choice(f1);
  stream.setLabel(f0, stream.set("*/"));
  stream.call(term);
  stream.partialCommit(f0);
  stream.setLabel(f1, stream.captureEnd());
  stream.ret();

  // Term
  auto t0 = stream.label();
  auto t1 = stream.label();
  stream.setLabel(term, stream.captureBegin(12));
  stream.choice(t0);
  stream.call(number);
  stream.commit(t1);
  stream.setLabel(t0, stream.match('('));
  stream.call(exp);
  stream.match(')');
  stream.setLabel(t1, stream.captureEnd());
  stream.ret();

  // Number
  auto n0 = stream.label();
  auto n1 = stream.label();
  stream.setLabel(number, stream.captureBegin(13));
  stream.set("1234567890");

  stream.choice(n1);
  stream.setLabel(n0, stream.set("1234567890"));
  stream.partialCommit(n0);
  stream.setLabel(n1, stream.captureEnd());
  stream.ret();

  auto program = stream.finalize();
  print(std::cerr, program);
  // const uint8_t sp[] = "1*2+3/4";
  const uint8_t sp[] = "(1+2)*(3-4)/(5+6)";
  auto *se = sp + sizeof(sp);
  std::cerr << "sp=" << (void *)sp << ", se=" << (void *)se << "\n";
  std::vector<Node *> captures;
  std::vector<Node *> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  std::cerr << "test results" << std::endl;
  print(std::cerr, captures);
  std::cerr << std::endl;
  EXPECT_TRUE(result);
}

TEST(Machine, Memoization) {
  InsnStream stream;

  auto exp = stream.label();
  auto factor = stream.label();
  auto term = stream.label();
  auto number = stream.label();

  // Top level
  stream.call(exp);
  stream.end();

  // Expr
  auto e0 = stream.label();
  auto e1 = stream.label();
  auto eEnd = stream.label();
  stream.setLabel(exp, stream.memoOpen(eEnd, 10));
  stream.captureBegin(10);
  stream.call(factor);
  stream.choice(e1);
  stream.setLabel(e0, stream.set("+-"));
  stream.call(factor);
  stream.partialCommit(e0);
  stream.setLabel(e1, stream.captureEnd());
  stream.memoClose();
  stream.setLabel(eEnd, stream.ret());

  // Factor
  auto f0 = stream.label();
  auto f1 = stream.label();
  stream.setLabel(factor, stream.captureBegin(11));
  stream.call(term);
  stream.choice(f1);
  stream.setLabel(f0, stream.set("*/"));
  stream.call(term);
  stream.partialCommit(f0);
  stream.setLabel(f1, stream.captureEnd());
  stream.ret();

  // Term
  auto t0 = stream.label();
  auto t1 = stream.label();
  stream.setLabel(term, stream.captureBegin(12));
  stream.choice(t0);
  stream.call(number);
  stream.commit(t1);
  stream.setLabel(t0, stream.match('('));
  stream.call(exp);
  stream.match(')');
  stream.setLabel(t1, stream.captureEnd());
  stream.ret();

  // Number
  auto n0 = stream.label();
  auto n1 = stream.label();
  stream.setLabel(number, stream.captureBegin(13));
  stream.set("1234567890");

  stream.choice(n1);
  stream.setLabel(n0, stream.set("1234567890"));
  stream.partialCommit(n0);
  stream.setLabel(n1, stream.captureEnd());
  stream.ret();

  auto program = stream.finalize();
  print(std::cerr, program);
  // const uint8_t sp[] = "1*2+3/4";
  const uint8_t sp[] = "(1+2)";
  auto *se = sp + sizeof(sp);
  std::cerr << "sp=" << (void *)sp << ", se=" << (void *)se << "\n";

  std::vector<Node *> captures;
  MemoTable memoTable;
  std::vector<Node *> diagnostics;
  auto result =
      Machine::parse(program, memoTable, sp, se, captures, diagnostics);
  EXPECT_TRUE(result);
}

TEST(Machine, MemoFail) {
  InsnStream stream;
  auto l0 = stream.label();
  stream.memoOpen(l0, 0);
  stream.match('a');
  stream.memoClose();
  stream.setLabel(l0, stream.end());
  auto program = stream.finalize();

  const uint8_t sp[] = "b";
  auto *se = sp + sizeof(sp);

  MemoTable memoTable;
  std::vector<Node *> captures;
  std::vector<Node *> diagnostics;
  Machine::parse(program, memoTable, sp, se, captures, diagnostics);
}
#endif

TEST(Parser, MyGrammar) {
  InsnStream s;

  auto ws = s.label();
  auto num = s.label();
  auto id = s.label();
  auto atom = s.label();
  auto expression = s.label();
  auto group = s.label();
  auto declaration = s.label();
  auto definition = s.label();
  auto statement = s.label();
  auto file = s.label();

  enum CaptureId {
    IdId,
    NumId,
    ExprId,
    DeclId,
    DefId,
    StmtId,
    TrailingId,
  };
  using namespace circt::hwl::p;

  // clang-format off
  program(file,
    rule(ws, star(p::set(" \n\t"))),
    rule(id, capture(IdId, p::plus(p::set("abcdefghijklmnopqrstuvwxyz")))),
    rule(num, capture(NumId, p::plus(p::set("1234567890")))),
    rule(atom, alt(id, num, group)),
    rule(expression, capture(ExprId, atom, star(ws, atom))),
    rule(group,
      "(",
      require("require expression inside group", expression),
      require("missing closing parenthesis", ")")),
    rule(declaration, capture(DeclId, 
      id, ws, ":", ws,
      require("missing type expression", expression), ws)),
    rule(definition, capture(DefId,
      p::plus(id), ws, "=", ws, expression)),
    rule (statement, capture(StmtId,
      alt(definition, declaration),
      ws,
      require("missing semicolon", ";"))),
    rule(file, star(ws, statement), ws, require("expected declaration or definition", p::failIf(p::any()))))(s);
  // clang-format on
  auto program = s.finalize();

  // const uint8_t sp[] = "a : foo bar baz; dah : dah;";
  const uint8_t sp[] = "";
  auto *se = sp + sizeof(sp);

  std::vector<Node *> captures;
  std::vector<Diagnostic> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  // print(std::cerr, captures);
  //  std::cerr << "\n";
  //  for (auto &d : diagnostics) {
  //    std::cerr << (d.sp - sp) << ": " << d.message << "\n";
  //  }
  EXPECT_TRUE(result);
  print(std::cout, program);
}
