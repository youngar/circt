#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/Parse/Machine.h"
#include "gtest/gtest.h"

using namespace std;
using namespace circt;
using namespace circt::hwml;

static size_t verify(std::ostream &out, MemoNode *node) {
  auto leftHeight = 0;
  auto rightHeight = 0;
  auto maxExamined = node->getExamined();
  if (auto *l = node->getLeftChild()) {
    leftHeight = verify(out, l);
    maxExamined = std::max(maxExamined, l->getMaxExamined());
    assert(l->getStart() <= node->getStart());
    if (l->getStart() == node->getStart())
      assert(l->getId() < node->getId());
  }

  if (auto *r = node->getRightChild()) {
    rightHeight = verify(out, r);
    maxExamined = std::max(maxExamined, r->getMaxExamined());
    assert(r->getStart() >= node->getStart());
    if (r->getStart() == node->getStart())
      assert(r->getId() > node->getId());
  }

  if (node->getMaxExamined() != maxExamined) {
    out << "balance " << node->getBalance() << " should be "
        << (rightHeight - leftHeight) << "\n";
    out << "maxExamined " << node->getMaxExamined() << " should be "
        << maxExamined << "\n";
    MemoTable().dump(std::cout, node);
    out << *node << "\n";
  }

  assert(node->getBalance() == (rightHeight - leftHeight));
  assert(node->getMaxExamined() == maxExamined);
  return std::max(leftHeight, rightHeight) + 1;
}

void verify(std::ostream &out, MemoTable &tree) {
  if (tree.root)
    verify(out, tree.root);
}

TEST(Machine, Match) {
  InsnStream stream;
  stream.match('a');
  stream.end();
  auto program = stream.finalize();

  const uint8_t sp[] = "a";
  auto *se = sp + sizeof(sp);
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
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
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
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

TEST(Machine, CaptureReduce) {
  InsnStream stream;
  // expr = :: (expr '+' "a") / "a"
  std::cout << "AAAAAAAAAAAAA" << std::endl;

  auto term = stream.label();
  auto id = stream.label();
  auto ws = stream.label();

  // EXP
  stream.call(ws);
  stream.captureBegin(5);
  stream.call(term);
  stream.captureEnd();
  stream.call(ws);

  stream.star([&](InsnStream &stream) {
    stream.captureBegin(4);
    stream.captureBegin(2);
    stream.set("+-");
    stream.captureEnd();
    stream.call(ws);
    stream.call(term);
    stream.captureEndReduce();
  });
  stream.end();

  stream.setLabel(term, stream.captureBegin(3));
  stream.call(id);
  stream.call(ws);
  stream.star([&](InsnStream &stream) {
    stream.captureBegin(2);
    stream.set("*/");
    stream.captureEnd();
    stream.call(ws);
    stream.call(term);
    stream.call(ws);
  });
  stream.captureEnd();
  stream.ret();

  // ID
  stream.setLabel(id, stream.captureBegin(1));
  stream.plus([&](InsnStream &stream) { stream.set("abcdefgh"); });
  stream.captureEnd();
  stream.ret();

  // WS
  stream.setLabel(ws,
                  stream.star([&](InsnStream &stream) { stream.set(" "); }));
  stream.ret();

  auto program = stream.finalize();
  print(std::cerr, program);
  // const uint8_t sp[] = "1*2+3/4";
  const uint8_t sp[] = "a * a * a + b / b / b - c * c * c";
  auto *se = sp + sizeof(sp);
  std::cerr << "sp=" << (void *)sp << ", se=" << (void *)se << "\n";
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  std::cerr << "test results" << std::endl;
  print(std::cerr, captures);
  std::cerr << std::endl;
  EXPECT_TRUE(result);
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
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
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

  std::vector<Capture> captures;
  MemoTable memoTable;
  std::vector<Diagnostic> diagnostics;
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
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
  Machine::parse(program, memoTable, sp, se, captures, diagnostics);
}

MemoNode *insert(MemoTable &tree, Position offset, size_t length = 1,
                 size_t examined = 1) {
  return tree.insert(123, offset, length, examined, {});
}

TEST(MemoTable, Tree) {
  const char *input = "foo bar baz";
  auto sp = (Position)input;
  MemoTable tree;
  verify(std::cout, tree);
  tree.insert(0, sp + 1, 1, 1, {});
  verify(std::cout, tree);
  tree.insert(0, sp + 2, 1, 1, {});
  verify(std::cout, tree);
  tree.insert(0, sp + 0, 1, 1, {});
  verify(std::cout, tree);
}

TEST(MemoTable, Empty) {
  MemoTable tree;
  verify(std::cout, tree);
  tree.invalidate(0, 0, 1);
  verify(std::cout, tree);
}

TEST(MemoTable, Root) {
  MemoTable tree;
  verify(std::cout, tree);
  auto c = insert(tree, 0);
  EXPECT_EQ(tree.root, c);
  verify(std::cout, tree);
  tree.invalidate(0, 0, 1);
  EXPECT_EQ(tree.root, nullptr);
  verify(std::cout, tree);
}

/// Take a right-heavy tree, invalidate the left side, and cause
/// the right side to rotate leftward into root.
TEST(MemoTable, RemoveLeft) {
  MemoTable tree;
  auto c = insert(tree, 1);
  verify(std::cout, tree);
  auto l = insert(tree, 0);
  verify(std::cout, tree);
  auto r = insert(tree, 2);
  verify(std::cout, tree);
  auto rr = insert(tree, 3);
  verify(std::cout, tree);

  //     C=1
  //    X    X
  // L=0(D)  R=2
  //            X
  //             RR=3
  //
  // Delete L. R rotates to center, C becom
  EXPECT_EQ(tree.root, c);
  EXPECT_EQ(c->getLeftChild(), l);
  EXPECT_EQ(l->getLeftChild(), nullptr);
  EXPECT_EQ(l->getRightChild(), nullptr);
  EXPECT_EQ(c->getRightChild(), r);
  EXPECT_EQ(r->getLeftChild(), nullptr);
  EXPECT_EQ(r->getRightChild(), rr);
  EXPECT_EQ(rr->getLeftChild(), nullptr);
  EXPECT_EQ(rr->getRightChild(), nullptr);
  std::cout << "!!! dumping tree before invalidate";
  tree.dump(std::cout);

  tree.invalidate(0, 0, 1);
  std::cout << "!!! dumping tree after invalidate";
  tree.dump(std::cout);
  verify(std::cout, tree);

  // R
  // |-C
  // `-RR
  EXPECT_EQ(tree.root, r);
  EXPECT_EQ(r->getLeftChild(), c);
  EXPECT_EQ(r->getRightChild(), rr);
}

TEST(MemoTable, RemoveRoot) {
  //         2
  //     1       4
  //  0        3   5
  //                 6
  //

  //        4
  //    2      5
  //  1   3      6
  // 0

  //      C=2  <-- delete
  //     /    |
  //   L=1    R=4
  //  /     /    |
  // LL=0  RL=3  RR=5
  //                 |
  //                 RRR=6
  //
  // Delete C.  RL should replace C in the tree, as the smallest element of
  // the R subtree.  This will cause R to become unbalanced.
  //
  //     RL=3
  //    /   |
  //  L=1    RR=5
  //         /   |
  //       R=4   RRR=6
  MemoTable tree;
  auto c = insert(tree, 2);
  verify(std::cout, tree);
  auto l = insert(tree, 1);
  verify(std::cout, tree);
  auto r = insert(tree, 4);
  verify(std::cout, tree);
  auto ll = insert(tree, 0);
  verify(std::cout, tree);
  auto rl = insert(tree, 3);
  verify(std::cout, tree);
  auto rr = insert(tree, 5);
  verify(std::cout, tree);
  auto rrr = insert(tree, 6);
  verify(std::cout, tree);

  EXPECT_EQ(tree.root, c);
  EXPECT_EQ(c->getLeftChild(), l);
  EXPECT_EQ(l->getLeftChild(), ll);
  EXPECT_EQ(ll->getLeftChild(), nullptr);
  EXPECT_EQ(ll->getRightChild(), nullptr);
  EXPECT_EQ(l->getRightChild(), nullptr);
  EXPECT_EQ(c->getRightChild(), r);
  EXPECT_EQ(r->getLeftChild(), rl);
  EXPECT_EQ(rl->getLeftChild(), nullptr);
  EXPECT_EQ(rl->getRightChild(), nullptr);
  EXPECT_EQ(r->getRightChild(), rr);
  EXPECT_EQ(rr->getLeftChild(), nullptr);
  EXPECT_EQ(rr->getRightChild(), rrr);
  EXPECT_EQ(rrr->getLeftChild(), nullptr);
  EXPECT_EQ(rrr->getRightChild(), nullptr);

  tree.dump(std::cout);
  tree.invalidate(1, 0, 1);
  tree.dump(std::cout);
  verify(std::cout, tree);
}

#include "circt/HWML/Parse/Insn.h"
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
  using namespace circt::hwml::p;
  // clang-format off
  p::program(file,
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
    rule(file, star(ws, statement), ws, require("expected declaration or definition", p::failIf(p::any())))
  )(s);

  // clang-format on
  p::program(file, rule(file, p::any(1)))(s);
  auto program = s.finalize();

  //const uint8_t sp[] = "a : foo bar baz; dah : dah;";
  const uint8_t sp[] = "";
  auto *se = sp + sizeof(sp);
  std::vector<Capture> captures;
  std::vector<Diagnostic> diagnostics;
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  std::cerr << "\n";
  for (auto &d : diagnostics) {
    std::cerr << (d.sp - sp) << ": " << d.message << "\n";
  }
  EXPECT_TRUE(result);
  print(std::cout, program);
}