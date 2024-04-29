#include "circt/HWML/Parse/MemoTable.h"
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
