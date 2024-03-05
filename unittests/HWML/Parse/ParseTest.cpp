#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/Parse/Machine.h"
#include "gtest/gtest.h"

using namespace std;
using namespace circt;
using namespace circt::hwml;

static size_t verify(std::ostream &out, MemoNode *node) {
  auto leftHeight = 0;
  auto rightHeight = 0;
  auto examinedMax = node->examined;
  if (auto *l = node->lchild) {
    leftHeight = verify(out, l);
    examinedMax = std::max(examinedMax, (l->sp - node->sp) + l->examined_max);
    assert(l->sp <= node->sp);
    if (l->sp == node->sp)
      assert(l->id < node->id);
  }

  if (auto *r = node->rchild) {
    rightHeight = verify(out, r);
    examinedMax = std::max(examinedMax, (r->sp - node->sp) + r->examined_max);
    assert(r->sp >= node->sp);
    if (r->sp == node->sp)
      assert(r->id > node->id);
  }

  // out << "balance " << node->balance << " should be "
  //     << (rightHeight - leftHeight);
  // out << "examinedMax " << node->examined_max << " should be " <<
  // examinedMax;
  assert(node->balance == (rightHeight - leftHeight));
  assert(node->examined_max == examinedMax);
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
  auto result = Machine::parse(program, memoTable, sp, se, captures);
  std::cerr << "test results" << std::endl;
  print(std::cerr, captures);
  std::cerr << std::endl;

  std::cerr << "**********\n";
  result = Machine::parse(program, memoTable, sp, se, captures);
  std::cerr << "test results" << std::endl;
  print(std::cerr, captures);
  std::cerr << std::endl;
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
  Machine::parse(program, memoTable, sp, se, captures);

  Machine::parse(program, memoTable, sp, se, captures);
}

MemoNode *insert(MemoTable &tree, size_t offset, size_t length = 1,
                 size_t examined = 1) {
  auto sp = (const uint8_t *)offset;
  return tree.insert(123, sp, length, examined, {});
}

TEST(MemoTable, Tree) {
  const char *input = "foo bar baz";
  auto sp = (const uint8_t *)input;
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
  EXPECT_EQ(c->lchild, l);
  EXPECT_EQ(l->lchild, nullptr);
  EXPECT_EQ(l->rchild, nullptr);
  EXPECT_EQ(c->rchild, r);
  EXPECT_EQ(r->lchild, nullptr);
  EXPECT_EQ(r->rchild, rr);
  EXPECT_EQ(rr->lchild, nullptr);
  EXPECT_EQ(rr->rchild, nullptr);

  tree.invalidate(0, 0, 1);
  tree.dump();
  verify(std::cout, tree);

  // R
  // |-C
  // `-RR
  EXPECT_EQ(tree.root, r);
  EXPECT_EQ(r->lchild, c);
  EXPECT_EQ(r->rchild, rr);
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
  EXPECT_EQ(c->lchild, l);
  EXPECT_EQ(l->lchild, ll);
  EXPECT_EQ(ll->lchild, nullptr);
  EXPECT_EQ(ll->rchild, nullptr);
  EXPECT_EQ(l->rchild, nullptr);
  EXPECT_EQ(c->rchild, r);
  EXPECT_EQ(r->lchild, rl);
  EXPECT_EQ(rl->lchild, nullptr);
  EXPECT_EQ(rl->rchild, nullptr);
  EXPECT_EQ(r->rchild, rr);
  EXPECT_EQ(rr->lchild, nullptr);
  EXPECT_EQ(rr->rchild, rrr);
  EXPECT_EQ(rrr->lchild, nullptr);
  EXPECT_EQ(rrr->rchild, nullptr);

  tree.invalidate((uint8_t *)1, 0, 1);
  verify(std::cout, tree);
}
