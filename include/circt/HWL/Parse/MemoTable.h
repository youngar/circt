#ifndef CIRCT_HWL_PARSE_MEMOTABLE_H
#define CIRCT_HWL_PARSE_MEMOTABLE_H

#include "circt/HWL/Parse/Capture.h"
#include "circt/HWL/Parse/Diagnostic.h"
#include "circt/HWL/Parse/Insn.h"
#include "circt/Support/LLVM.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdint.h>
#include <vector>

namespace circt {
namespace hwl {

enum class Order { LT = -1, EQ = 0, GT = +1 };

/// A diagnostic message produced by the parser.
struct Diagnostic {
  Diagnostic(const std::string &message, Position position)
      : message(message), position(position) {}

  /// Get the message of this diagnostic.
  std::string getMessage() const { return message; }

  /// Get this position of this diagnostic.
  Position getPosition() const { return position; }

  /// Set this position of this diagnostic.
  void setPosition(Position position) { this->position = position; }

  /// Shift the position of this diagnostic.
  void shift(signed delta) { setPosition(getPosition() + delta); }

private:
  std::string message;
  Position position;
};

/// Base class for the parser's captures, diagnostics, and memoization entries.
struct Node {
  Node(mlir::TypeID typeID) : typeID(typeID) {}

  mlir::TypeID getTypeID() const { return typeID; }

protected:
  mlir::TypeID typeID;
};

struct Capture;
struct MemoEntry;

template <typename Derived>
struct NodeMixin : public Node {
  NodeMixin() : Node(mlir::TypeID::get<Derived>()) {}
  static bool classof(const Node *node) {
    return node->getTypeID() == mlir::TypeID::get<Derived>();
  }
};

/// Text captured by the parser.  A capture may have subcaptures.
struct Capture : public NodeMixin<Capture> {
  Capture(std::uintptr_t id, Position position, std::size_t size,
          std::vector<Node *> &&children)
      : NodeMixin() {}

  /// Get the capture ID.
  std::uintptr_t getId() const { return id; }

  /// Get the size of this capture.
  std::size_t getSize() const { return size; }

  /// Get this position of this diagnostic.
  Position getPosition() const { return position; }

  /// Set this position of this diagnostic.
  void setPosition(Position position) { this->position = position; }

  /// Get the list of subcaptures.
  std::vector<Node *> getChildren() { return children; }

private:
  std::uintptr_t id;
  Position position;
  std::size_t size;
  std::vector<Node *> children;
};

/// An entry in the memoization table. A memoization entry contains the any
/// captures and diagnostics
struct MemoEntry : public NodeMixin<MemoEntry> {
  MemoEntry(size_t length, size_t examinedLength,
            std::vector<Node *> &&captures,
            std::vector<Diagnostic> &&diagnostics)
      : NodeMixin(), length(length), examinedLength(examinedLength),
        captures(std::move(captures)), diagnostics(std::move(diagnostics)) {}

  ~MemoEntry() {
    // Helper to free all capture recursively, until we hit another memo entry.
    std::function<void(Node *)> free = [&](Node *node) {
      if (auto *capture = dyn_cast<Capture>(node)) {
        for (auto *child : capture->getChildren())
          free(child);
        delete capture;
      }
    };
    for (auto *capture : captures)
      free(capture);
  }

  /// Get the number of characters matched by this entry.
  size_t getLength() const { return length; }

  /// The number of bytes examined by this entry. This can be larger than the
  /// length if the entry was constructed by a pattern that utilized lookahead.
  size_t getExaminedLength() const { return examinedLength; }

  /// Get the top-level captures under this entry.
  const std::vector<Node *> &getCaptures() const { return captures; }

  /// Get the diagnostics created under this entry.
  const std::vector<Diagnostic> &getDiagnostics() const { return diagnostics; }

  /// Shift the position of any memoized data, including captures and
  /// diagnostics by delta. This is used when text is inserted or deleted before
  /// this node in the buffer so that we can reuse the memoized results.
  void shift(signed delta) {
    // Helper to recursively shift MemoEntry's.
    std::function<void(Node *, signed)> shift = [&](Node *node, signed delta) {
      if (auto *capture = dyn_cast<Capture>(node)) {
        capture->setPosition(capture->getPosition() + delta);
        for (auto *child : capture->getChildren())
          shift(child, delta);
      }
    };

    for (auto *capture : captures)
      shift(capture, delta);
    for (auto diagnostic : diagnostics)
      diagnostic.shift(delta);
  }

private:
  /// The number of character matched by this entry.
  size_t length;

  /// The number of character examined to match this entry.
  size_t examinedLength;

  /// The top-level captures for this memoization node.
  std::vector<Node *> captures;

  /// A list of all diagnostics create under this memoization.
  std::vector<Diagnostic> diagnostics;
};

template <typename T>
T &operator<<(T &os, const MemoEntry &entry) {
  os << "MemoEntry{";
  os << "length=" << entry.getLength() << ", ";
  os << "examinedLength=" << entry.getExaminedLength();
  os << "}";
  return os;
}

/// A node in the memoization tree.  Each node in the tree should have a unique
/// (capture id, start position) pair which is their key.  It contains a list of
/// entries which all share the same start point as the node. The memo entries
/// are assumed to be added or stored from smallest to largest, which should be
/// properly enforced by the parser.
struct MemoNode {
  MemoNode(RuleId id, Position start) : id(id), start(start) {}

  /// Get the identifier of the current memoization entry.
  RuleId getId() const { return id; }

  /// Get the start position of the current memoization entry.
  Position getStart() const { return start; }

  /// Set the start position of the memoization entries.
  void setStart(Position start) { this->start = start; }

  /// Returns true if this node has no memoization entries, and false otherwise.
  bool empty() const { return entries.empty(); }

  /// Get the length of the current memoization entry.
  size_t getLength() const { return entries.back().getLength(); }

  /// Get the examined length of the current memoization entry.
  size_t getExaminedLength() const {
    return entries.back().getExaminedLength();
  }

  /// Get the examined  of the current memoization entry.
  size_t getExamined() const { return getStart() + getExaminedLength(); }

  /// Get the captures of the current memoization entry.
  const std::vector<Node *> &getCaptures() const {
    return entries.back().getCaptures();
  }

  /// Get the diagnostics of the current memoization entry.
  const std::vector<Diagnostic> &getDiagnostics() const {
    return entries.back().getDiagnostics();
  }

  /// Get the maximum examined length of this node and all children nodes.  This
  /// can be used to search for nodes which overlap with a certain range.
  size_t getMaxExaminedLength() const { return maxExaminedLength; }

  /// Set the maximumed examined length of this node.
  void setMaxExaminedLength(size_t maxExaminedLength) {
    this->maxExaminedLength = maxExaminedLength;
  }

  /// Get the position corresponding to the getStart() + getMaxExaminedLength().
  Position getMaxExamined() const { return start + getMaxExaminedLength(); }

  /// Set the position corresponding to the getStart() + getMaxExaminedLength().
  void setMaxExamined(Position maxExamined) {
    setMaxExaminedLength(maxExamined - getStart());
  }

  /// Get the current balance of this node, which represents the relative height
  /// of the left and right subtrees. An AVL tree never allows the height of the
  /// left and right subtree to differ by more than 1, so the balance must be in
  /// the inclusive range [-1, 1]. -1 corresponds to left heavy, 0 corresponds
  /// to balanced, and 1 corresponds to a right heavy.
  signed getBalance() const { return balance; }

  /// Set the current balance of this node.
  void setBalance(signed balance) { this->balance = balance; }

  MemoNode *&getLeftChild() { return lchild; }
  MemoNode *getLeftChild() const { return lchild; }
  void setLeftChild(MemoNode *child) { lchild = child; }

  MemoNode *&getRightChild() { return rchild; }
  MemoNode *getRightChild() const { return rchild; }
  void setRightChild(MemoNode *child) { lchild = child; }

  /// Update this memo node with a new, larger match. The old match is saved so
  /// that if the larger memo entry is invalidated by an edit, then we can
  /// revert to a smaller entry at the same position.
  void update(size_t length, size_t examinedLength,
              std::vector<Node *> &&captures,
              std::vector<Diagnostic> &&diagnostics) {
#ifndef NDEBUG
    if (!entries.empty()) {
      const auto &e = entries.back();
      assert(e.getLength() < length);
      assert(e.getExaminedLength() < examinedLength);
    }
#endif
    entries.emplace_back(length, examinedLength, std::move(captures),
                         std::move(diagnostics));
    if (maxExaminedLength < examinedLength)
      maxExaminedLength = examinedLength;
  }

  /// Shift the position of any memoized data, including captures and
  /// diagnostics by delta. This is used when text is inserted or deleted before
  /// this node in the buffer so that we can reuse the memoized results.
  void shift(signed delta) {
    // Update the start position of this node.
    setStart(getStart() + delta);

    for (auto &entry : entries)
      entry.shift(delta);
  }

  /// Discard any memo entries which are longer than or equal to the length.
  [[nodiscard]] bool invalidate(size_t length) {
    while (!entries.empty()) {
      if (length <= entries.back().getExaminedLength())
        entries.pop_back();
      else
        return false;
    }
    return true;
  }

  /// Compare two nodes, first by the start position, then by capture
  /// identifier.
  Order compare(RuleId id, Position start) const {
    if (this->start < start)
      return Order::LT;
    if (this->start > start)
      return Order::GT;
    if (this->id < id)
      return Order::LT;
    if (this->id > id)
      return Order::GT;
    return Order::EQ;
  }

  /// Replace the memoization entries in this node with the entries of another
  /// node.
  void assignMemoEntries(MemoNode &&rhs) {
    assert(empty() && "MemoNode must be empty to be assigned");
    id = rhs.id;
    start = rhs.start;
    maxExaminedLength = rhs.maxExaminedLength;
    entries = std::move(rhs.entries);
  }

  template <typename T>
  friend T &operator<<(T &os, const MemoNode &node);

private:
  // Lookup keys.
  RuleId id;
  Position start;

  // The memoization entries.
  std::vector<MemoEntry> entries;

  // Tree metadata.
  size_t maxExaminedLength = 0;
  signed balance = 0;
  MemoNode *lchild = nullptr;
  MemoNode *rchild = nullptr;
};

template <typename T>
T &operator<<(T &os, const MemoNode &node) {
  os << "{";
  os << "id='" << node.getId() << "', ";
  os << "start=" << node.getStart() << ", ";
  os << "examined_max=" << node.getMaxExaminedLength() << ", ";
  os << "balance=" << node.getBalance() << "}";
  os << "entries=[";
  for (auto &entry : node.entries)
    os << entry;
  os << "]";
  return os;
}

using Path = std::vector<MemoNode **>;

/// A guard which, on destruction, returns the path to it's original size at
/// the time of construction.
struct PathGuard {
  PathGuard(Path &path) : path(path), size(path.size()) {}
  ~PathGuard() { path.resize(size); }

private:
  Path &path;
  size_t size;
};

struct MemoTable {

  MemoNode **find(MemoNode **slot, RuleId id, Position start) {
    while (*slot) {
      auto *node = *slot;
      auto order = node->compare(id, start);
      if (order == Order::LT)
        slot = &node->getLeftChild();
      else if (order == Order::GT)
        slot = &node->getRightChild();
      else
        return slot;
    }
    return nullptr;
  }

  MemoNode **find(RuleId id, Position start) { return find(&root, id, start); }

  MemoNode *lookup(RuleId id, Position sp) {
    auto *slot = find(id, sp);
    if (!slot)
      return nullptr;
    return *slot;
  }

  void free(MemoNode *node) {
    if (node->getLeftChild())
      free(node->getLeftChild());
    if (node->getRightChild())
      free(node->getRightChild());
    delete node;
  }

  size_t height(MemoNode *node) {
    if (!node)
      return 0;
    return 1 + std::max(height(node->getLeftChild()),
                        height(node->getRightChild()));
  }

  /// Follows left pointers until it can go no further, returning the
  /// smallest element in the tree.  Has an output path vector. Returns a
  /// pointer to the slot, which contains a pointer to the smallest element.
  MemoNode **findSuccessor(MemoNode **slot,
                           std::vector<MemoNode **> &path) const {
    assert(slot && *slot);
    while ((*slot)->getLeftChild()) {
      path.emplace_back(slot);
      slot = &(*slot)->getLeftChild();
    }
    return slot;
  }

  /// Propagate the examined_max from the children of the node to the node
  /// itself.
  bool propMaxExamined(MemoNode *parent) {
    auto maxExamined = parent->getExamined();
    if (auto *leftChild = parent->getLeftChild()) {
      if (leftChild->getMaxExamined() > maxExamined)
        maxExamined = leftChild->getMaxExamined();
    }
    if (auto *rightChild = parent->getRightChild()) {
      if (rightChild->getMaxExamined() > maxExamined)
        maxExamined = rightChild->getMaxExamined();
    }
    if (maxExamined == parent->getMaxExamined())
      return false;
    // Return true if updated.
    parent->setMaxExamined(maxExamined);
    return true;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Rotations
  //////////////////////////////////////////////////////////////////////////////

  ///       C
  ///   ┌───┴───┐
  ///   L       R
  /// ┌─┴─┐   ┌─┴─┐
  /// LL  LR  RL  RR
  ///
  /// Rotating right at C gives:
  ///
  ///       L
  ///   ┌───┴───┐
  ///   LL      C
  ///       ┌───┴───┐
  ///       LR      R
  ///             ┌─┴─┐
  ///             RL  RR
  MemoNode *rotateR(MemoNode *root) {
    auto node = root->getLeftChild();
    root->getLeftChild() = node->getRightChild();
    node->getRightChild() = root;
    if (node->getBalance() == 0) {
      root->setBalance(-1);
      node->setBalance(+1);
    } else {
      root->setBalance(0);
      node->setBalance(0);
    }
    node->setMaxExaminedLength(root->getMaxExaminedLength());
    propMaxExamined(root);
    propMaxExamined(node);
    return node;
  }

  ///           C
  ///       ┌───┴───┐
  ///       L       R
  ///     ┌─┴─┐   ┌─┴─┐
  ///     LL  LR  RL  RR
  ///
  /// Rotating left at C gives:
  ///
  ///           R
  ///       ┌───┴───┐
  ///       C      RR
  ///   ┌───┴───┐
  ///   L       RL
  /// ┌─┴─┐
  /// LL  LR
  MemoNode *rotateL(MemoNode *root) {
    auto *node = root->getRightChild();
    root->getRightChild() = node->getLeftChild();
    node->getLeftChild() = root;
    if (node->getBalance() == 0) {
      root->setBalance(+1);
      node->setBalance(-1);
    } else {
      root->setBalance(0);
      node->setBalance(0);
    }
    propMaxExamined(root);
    propMaxExamined(node);
    return node;
  }

  /// Perform a right rotation followed by a left rotation.
  MemoNode *rotateRL(MemoNode *root) {

    // Assumption: the root node has balance +2.
    auto *l = root;
    auto *r = root->getRightChild();
    auto *c = root->getRightChild()->getLeftChild();
    auto *cl = c->getLeftChild();
    auto *cr = c->getRightChild();
    assert(r->getBalance() == -1);

    // Rotate center and right node to the right.
    r->getLeftChild() = cr;
    c->getRightChild() = r;

    // Rotate center and left node to the left.
    l->getRightChild() = cl;
    c->getLeftChild() = l;

    // Fix up balance.
    if (c->getBalance() == 0) {
      l->setBalance(0);
      r->setBalance(0);
    } else if (c->getBalance() > 0) {
      l->setBalance(-1);
      c->setBalance(0);
      r->setBalance(0);
    } else {
      l->setBalance(0);
      c->setBalance(0);
      r->setBalance(1);
    }

    propMaxExamined(l);
    propMaxExamined(r);
    propMaxExamined(c);

    // center node is the new root.
    return c;
  }

  MemoNode *rotateLR(MemoNode *root) {
    // Assumption: the root node has balance +2.
    auto r = root;
    auto l = root->getLeftChild();
    auto c = root->getLeftChild()->getRightChild();
    auto cl = c->getLeftChild();
    auto cr = c->getRightChild();
    assert(l->getBalance() == +1);

    // Rotate center and left node to the left.
    l->getRightChild() = cl;
    c->getLeftChild() = l;

    // Rotate center and right node to the right.
    r->getLeftChild() = cr;
    c->getRightChild() = r;

    // Fix up balance.
    if (c->getBalance() == 0) {
      l->setBalance(0);
      r->setBalance(0);
    } else if (c->getBalance() > 0) {
      l->setBalance(-1);
      c->setBalance(0);
      r->setBalance(0);
    } else {
      l->setBalance(0);
      c->setBalance(0);
      r->setBalance(1);
    }

    propMaxExamined(l);
    propMaxExamined(r);
    propMaxExamined(c);

    // Center node is the new root.
    return c;
  }

  /// The tree rooted by `node` has a temporary balance of -2 (left heavy).
  /// Rebalance the tree by rotating rightwards, and return the new root.
  MemoNode *rebalanceLHeavy(MemoNode *node) {
    assert(node->getBalance() == -1);
    if (node->getLeftChild()->getBalance() == -1)
      return rotateLR(node);
    return rotateR(node);
  }

  /// The tree rooted by `node` has a temporary balance of +2 (right heavy).
  /// Rebalance the tree by rotating leftwards, and return the new root.
  MemoNode *rebalanceRHeavy(MemoNode *node) {
    assert(node->getBalance() == +1);
    if (node->getRightChild()->getBalance() == -1)
      return rotateRL(node);
    return rotateL(node);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Insertion
  //////////////////////////////////////////////////////////////////////////////

  /// Retrace through the ancestors and rotate as needed to ensure balance.
  void rebalanceInsertion(MemoNode **slot,
                          const std::vector<MemoNode **> &ancestors) {
    auto i = ancestors.rbegin();
    auto e = ancestors.rend();

    // Propagate the max from the child to the parent.
    auto propMaxExamined = [](MemoNode *child, MemoNode *parent) {
      if (child->getMaxExamined() > parent->getMaxExamined()) {
        parent->setMaxExaminedLength(child->getMaxExamined() -
                                     parent->getStart());
        return true;
      }
      return false;
    };

    // Propagate the max from the current element of the path all the way up to
    // the top.
    auto propMaxExaminedToTop = [&]() {
      for (; i != e; ++i) {
        auto **parentSlot = *i;
        // Propagate the max examined up one level.  If it did not change, then
        // we don't have to propagate it further. We only have to check the
        // child which grew has caused our height to change.
        if (!propMaxExamined(*slot, *parentSlot))
          return;
        slot = parentSlot;
      }
    };

    for (; i != e; ++i) {
      auto **parentSlot = *i;
      auto *parent = *parentSlot;
      auto *node = *slot;

      if (slot == &parent->getLeftChild()) {
        // Left side grew.

        // Right-heavy => balanced.
        if (parent->getBalance() == +1) {
          parent->setBalance(0);
          propMaxExaminedToTop();
          return;
        }

        // Balanced => left-heavy.
        if (parent->getBalance() == 0) {
          parent->setBalance(-1);
          propMaxExamined(node, parent);
          slot = parentSlot;
          continue;
        }

        // Left-heavy => very-left-heavy.
        if (node->getBalance() == +1)
          *parentSlot = rotateLR(parent);
        else
          *parentSlot = rotateR(parent);
        slot = parentSlot;
      } else {
        // Right side grew.

        // Left-heavy => balanced.
        if (parent->getBalance() < 0) {
          parent->setBalance(0);
          propMaxExaminedToTop();
          return;
        }

        // Balanced => right-heavy
        if (parent->getBalance() == 0) {
          parent->setBalance(+1);
          propMaxExamined(node, parent);
          slot = parentSlot;
          continue;
        }

        // Right-Heavy => Very-Right-Heavy
        if (node->getBalance() < 0)
          *parentSlot = rotateRL(parent);
        else
          *parentSlot = rotateL(parent);
        slot = parentSlot;
      }
    }
  }

  MemoNode *insert(RuleId id, Position start, size_t length, size_t examined,
                   std::vector<Node *> &&captures,
                   std::vector<Diagnostic> &&diagnostics) {
    std::vector<MemoNode **> ancestorSlots;
    auto **slot = &root;
    // ancestorSlots.push_back(slot);
    auto *node = root;
    while (node) {
      auto order = node->compare(id, start);
      // if the node is _less_, go to the right slot.
      if (order == Order::LT) {
        ancestorSlots.push_back(slot);
        slot = &node->getRightChild();
        node = *slot;
        continue;
      }
      // if the node is _more_, go to the left slot.
      if (order == Order::GT) {
        ancestorSlots.push_back(slot);
        slot = &node->getLeftChild();
        node = *slot;
        continue;
      }
      // Otherwise the memo node already exists.
      node->update(length, examined, std::move(captures),
                   std::move(diagnostics));
      return node;
    }

    // Create a new node and insert it into the tree.
    node = new MemoNode(id, start);
    node->update(length, examined, std::move(captures), std::move(diagnostics));
    *slot = node;
    rebalanceInsertion(slot, ancestorSlots);
    return node;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Removal
  //////////////////////////////////////////////////////////////////////////////

  /// The tree rooted at `node` has changed height due to a deletion.
  /// `node` is already in AVL shape. Rebalance the ancestors by retracing
  /// through the `path` to the current node. At the point of calling:
  /// - We are rebalancing the parent of node.
  /// - We get the parent by looking at the first element of path.
  /// - The last element of path should be a slot in the grandparent which
  ///    points to the MemoNode which contains `slot`.
  void rebalanceRemoval(MemoNode **slot, const std::vector<MemoNode **> &path) {
    auto i = path.rbegin();
    auto e = path.rend();

    // Propagate the max from the current element of the path all the way up to
    // the top.
    auto propMaxExaminedToTop = [&]() {
      for (; i != e; ++i) {
        auto **parentSlot = *i;
        // Propagate the max examined up one level.  If it did not change, then
        // we don't have to propagate it further.
        if (!propMaxExamined(*parentSlot))
          return;
        slot = parentSlot;
      }
    };

    for (; i != e; ++i) {
      auto **parentSlot = *i;
      auto *parent = *parentSlot;

      // The left subtree shrunk.
      if (slot == &parent->getLeftChild()) {
        // Parent: balanced -> right-heavy.
        // Height is unchanged. Stop here.
        if (parent->getBalance() == 0) {
          parent->setBalance(+1);
          propMaxExaminedToTop();
          return;
        }

        // Parent: left-heavy -> balanced.
        // Height has decreased. Continue retracing the path.
        if (parent->getBalance() == -1) {
          parent->setBalance(0);
          propMaxExamined(parent);
          slot = parentSlot;
          continue;
        }

        // Parent: right-heavy -> very-right-heavy.
        // Rebalance the parent and continue retracing the path.
        *parentSlot = rebalanceRHeavy(parent);
        slot = parentSlot;
      }
      // The right subtree shrunk.
      else {
        // Parent: balanced -> left-heavy.
        // Height is unchanged. Stop here.
        if (parent->getBalance() == 0) {
          parent->setBalance(-1);
          propMaxExaminedToTop();
          return;
        }

        // Parent: right-heavy -> balanced.
        // Height has decreased. Continue retracing the path.
        if (parent->getBalance() == +1) {
          parent->setBalance(0);
          propMaxExamined(parent);
          slot = parentSlot;
          continue;
        }

        // Parent: left-heavy => very-left-heavy.
        // Rebalance the parent and continue retracing the path.
        *parentSlot = rebalanceLHeavy(parent);
        slot = parentSlot;
      }
    }
  }

  /// Delete the node who is pointed-at by the given slot. The path should
  /// contain all the nodes from the root node down to the target slot. The path
  /// is used during removal but will be reset before returning.
  void remove(MemoNode **slot, std::vector<MemoNode **> &path) {
    // Deletion is done by replacing our target with the smallest
    // child on the right. This effectively shrinks the tree on the right,
    // which needs to be rebalanced.
    PathGuard guard(path);
    auto *node = *slot;
    assert(node->empty() && "MemoNode must be empty to be removed");
    if (node->getLeftChild() && node->getRightChild()) {
      // We will replace the deleted node with its successor, which is the
      // least greater node in the tree.
      std::vector<MemoNode **> succPath;
      auto **succSlot = findSuccessor(&node->getRightChild(), succPath);
      auto *succ = *succSlot;
      // Detach the successor node from the tree, and replace it with it's
      // own getRightChild().
      *succSlot = succ->getRightChild();
      // Replace the deleted node with the successor, copying over its
      // children.
      node->assignMemoEntries(std::move(*succ));
      delete succ;
      // Removing the successor could require a rotation.
      rebalanceRemoval(succSlot, succPath);
    } else if (!node->getRightChild()) {
      *slot = node->getLeftChild();
    } else if (!node->getLeftChild()) {
      *slot = node->getRightChild();
    } else {
      // TODO: we are not deleting the removed node for some raison?
      *slot = nullptr;
    }

    rebalanceRemoval(slot, path);
  }

  void shift(MemoNode **slot, Position start, signed delta) {
    auto *node = *slot;
    if (node == nullptr)
      return;
    assert(!(node->getStart() <= start && start < node->getMaxExamined()) &&
           "node must not intersect insertion point");
    // If this is true, than then entire tree rooted at this node occurs before
    // the start.
    if (node->getMaxExamined() < start)
      return;

    if (node->getStart() >= start) {
      node->shift(delta);
      // If the current node is after the start, then the node to the left might
      // also be after the start.
      shift(&node->getLeftChild(), start, delta);
    }
    // Even if the current node is not after the start, the node to the right
    // might be after the start.
    shift(&node->getRightChild(), start, delta);
  }

  /// Remove cached memoization which fall in the range `[low, high)`.  If
  /// low == high, then no entries will be evicted.  Any remaining entries which
  /// are higher than the low position are shifted by delta.
  ///
  ///  Examples:
  ///
  void invalidate(MemoNode **slot, Position low, Position high, signed delta,
                  std::vector<MemoNode **> &path) {
    assert(slot && "slot pointer can not be null");
    auto *node = *slot;
    if (!node)
      return;

    // If this is true, than this node and its entire subtree occur before
    // the insertion point and can be skipped.
    if (node->getMaxExamined() <= low)
      return;

    // There are 5 possible cases to consider depending on how the memoization
    // nodes overlap with the invalid range.  For the center node, we have to
    // delete it if it overlaps with the invalid range, shift it if it starts
    // after the invalid range, and otherwise nothing.  For the left child you
    // have to potentially invalidate no matter what case it is.  For the right
    // child, we can guarantee that we only have to shift it if the current node
    // starts after the invalid range, but we potentially have to invalidate it
    // for the rest of the cases.
    //          |-invalid-|
    //    [1]  [2]  [3]  [4]  [5]
    // C:  n    d    d    d    s
    // L:  i    i    i    i    i
    // R:  i    i    i    i    s

    // The left child must always be visited, as although they will start before
    // the current node, they may have a larger examined max that causes them to
    // overlap with the invalid range.
    path.push_back(slot);
    invalidate(&node->getLeftChild(), low, high, delta, path);
    path.pop_back();

    // The node starts after the invalid range, and the right subtree is
    // guaranteed to start after the invalid range as well, so both must be
    // shifted.
    if (high <= node->getStart()) {
      node->setStart(node->getStart() + delta);
      shift(&node->getRightChild(), low, delta);
      // The left subtree might have shrunk, so we have to recalculate our max
      // examined.
      propMaxExamined(node);
      return;
    }

    // The right subtree needs to be invalidated.
    path.push_back(slot);
    invalidate(&node->getRightChild(), low, high, delta, path);
    path.pop_back();

    // propMaxExamined(node);

    // The current node falls in to the invalid range, so invalidate it. If
    // the node is empty afterwords, delete it. Otherwise, update its max
    // examined to reflect the removed entries.
    if (low < node->getExamined()) {
      if (node->invalidate(high - node->getStart())) {
        remove(slot, path);
        return;
      }
    }
    propMaxExamined(node);
  }

  void invalidate(Position position, signed inserted, signed removed) {
    if (!inserted && !removed)
      return;
    signed delta = inserted - removed;
    std::vector<MemoNode **> path;
    invalidate(&root, position, position + removed, delta, path);
  }

   void invalidateAll() {
     // TODO.
   }

  //////////////////////////////////////////////////////////////////////////////
  // Printing
  //////////////////////////////////////////////////////////////////////////////

  void dump(std::ostream &os, const std::vector<MemoNode **> &ancestors) const {
    auto i = ancestors.begin();
    auto e = ancestors.end();
    os << "[";
    bool first = true;
    while (i != e) {
      if (!first)
        os << ", ";
      os << "@" << *i << "=" << **i;
      first = false;
      ++i;
    }
    os << "]\n";
  }

  void dump(std::ostream &os, const char *label, const MemoNode *node,
            size_t depth) const {
    for (size_t i = 0; i < depth; ++i)
      os << "  ";
    os << label << ": ";
    if (!node) {
      os << "<nil>\n";
      return;
    }
    os << "@" << node << "=" << *node << "\n";
    dump(os, "L", node->getLeftChild(), depth + 1);
    dump(os, "R", node->getRightChild(), depth + 1);
  }

  void dump(std::ostream &os, const MemoNode *node, size_t depth = 0) const {
    dump(os, "C", node, depth);
  }

  void dump(std::ostream &os) const {
    os << "--------\n";
    dump(os, root);
    os << "--------\n";
  }

  MemoNode *root = nullptr;
};

} // namespace hwl
} // namespace circt
#endif
