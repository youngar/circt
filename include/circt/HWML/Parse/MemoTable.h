#ifndef CIRCT_HWML_PARSE_MEMOTABLE_H
#define CIRCT_HWML_PARSE_MEMOTABLE_H

#include "circt/HWML/Parse/Capture.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdint.h>
#include <vector>

namespace circt {
namespace hwml {

enum class Order { LT = -1, EQ = 0, GT = +1 };

struct MemoEntry {
  MemoEntry(size_t length, std::vector<Capture> &&captures)
      : length(length), captures(std::move(captures)) {}
  size_t length;
  std::vector<Capture> captures;
}

struct MemoNode {
  MemoNode(uintptr_t id, const uint8_t *sp, size_t length, size_t examined,
           std::vector<Capture> &&captures)
      : id(id), sp(sp), length(length), examined(examined),
        captures(std::move(captures)), examined_max(examined), balance(0),
        lchild(nullptr), rchild(nullptr) {}

  /// Compare two nodes, first by SP, then by ID.
  Order compare(uintptr_t rhs_id, const uint8_t *rhs_sp) const {
    if (sp < rhs_sp)
      return Order::LT;
    if (sp > rhs_sp)
      return Order::GT;
    if (id < rhs_id)
      return Order::LT;
    if (id > rhs_id)
      return Order::GT;
    return Order::EQ;
  }

  const uint8_t *examinedMaxEnd() const { return sp + examined_max; }
  const uint8_t *examinedEnd() const { return sp + examined; }
  // const uint8_t *end() const { return sp + length; }

  // Replace the memodata in this node with the memodata of another node.
  void assignMemoData(MemoNode &&rhs) {
    id = rhs.id;
    sp = rhs.sp;
    examined = rhs.examined;
    examined_max = rhs.examined_max;
    captures = std::move(rhs.captures);
  }

  // Lookup keys.
  uintptr_t id;
  const uint8_t *sp;

  // Cached values.
  size_t length;
  size_t examined;
  std::vector<Capture> captures;

  // Tree metadata.
  size_t examined_max;
  signed balance;
  MemoNode *lchild;
  MemoNode *rchild;
};

template <typename T>
T &operator<<(T &os, const MemoNode &node) {
  os << "{";
  os << "id='" << node.id << "', ";
  os << "sp=" << (void *)node.sp << ", ";
  os << "length=" << node.length << ", ";
  os << "examined=" << node.examined << ", ";
  os << "examined_max=" << (void *)node.examined_max << ", ";
  os << "balance=" << node.balance << "}";
  return os;
}

using Path = std::vector<MemoNode **>;

/// Guard which, on destruction, returns the path to it's original size at the
/// time of construction.
struct PathGuard {
  PathGuard(Path &path) : path(path), size(path.size()) {}
  ~PathGuard() { path.resize(size); }

  Path &path;
  size_t size;
};

struct MemoTable {
  MemoNode *root = nullptr;

  // Records that at position `offset`, delta bytes were added or removed.
  void edit(const uint8_t *position, size_t oldSize, size_t newSize) {}

  MemoNode **find(MemoNode **slot, uintptr_t id, const uint8_t *sp) {
    while (*slot) {
      auto *node = *slot;
      if (sp == node->sp) {
        if (id == node->id)
          return slot;
        else if (id < node->id)
          slot = &node->lchild;
        else
          slot = &node->rchild;
      } else if (sp < node->sp) {
        slot = &node->lchild;
      } else {
        slot = &node->rchild;
      }
    }
    return nullptr;
  }

  MemoNode **find(uintptr_t id, const uint8_t *sp) {
    return find(&root, id, sp);
  }

  MemoNode *lookup(uintptr_t id, const uint8_t *sp) {
    auto slot = find(id, sp);
    if (!slot)
      return nullptr;
    return *slot;
  }

  void free(MemoNode *node) {
    if (node->lchild)
      free(node->lchild);
    if (node->rchild)
      free(node->rchild);
    delete node;
  }

  size_t height(MemoNode *node) {
    if (!node)
      return 0;
    return 1 + std::max(height(node->lchild), height(node->rchild));
  }

  /// Follows left pointers until it can go no further, returning the smallest
  /// element in the tree.  Has an output path vector. Returns a pointer to
  /// the slot, which contains a pointer to the smallest element.
  MemoNode **findSuccessor(MemoNode **slot,
                           std::vector<MemoNode **> &path) const {
    assert(slot && *slot);
    while ((*slot)->lchild) {
      path.emplace_back(slot);
      slot = &(*slot)->lchild;
    }
    return slot;
  }

  /// Propagate the examined_max from the children of the node to the node
  /// itself.
  bool propMaxExamined(MemoNode *parent) {
    auto *examinedMax = parent->examinedMaxEnd();
    if (auto *lchild = parent->lchild) {
      if (lchild->examinedMaxEnd() > examinedMax)
        examinedMax = lchild->examinedMaxEnd();
    }
    if (auto *rchild = parent->rchild) {
      if (rchild->examinedMaxEnd() > examinedMax)
        examinedMax = rchild->examinedMaxEnd();
    }
    if (examinedMax == parent->examinedMaxEnd())
      return false;
    // Return true if updated.
    parent->examined_max = examinedMax - parent->sp;
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
    auto node = root->lchild;
    root->lchild = node->rchild;
    node->rchild = root;
    if (node->balance == 0) {
      root->balance = -1;
      node->balance = +1;
    } else {
      root->balance = 0;
      node->balance = 0;
    }
    node->examined_max = root->examined_max;
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
    auto *node = root->rchild;
    root->rchild = node->lchild;
    node->lchild = root;
    if (node->balance == 0) {
      root->balance = +1;
      node->balance = -1;
    } else {
      root->balance = 0;
      node->balance = 0;
    }
    propMaxExamined(root);
    propMaxExamined(node);
    return node;
  }

  MemoNode *rotateRL(MemoNode *root) {

    // Assumption: the root node has balance +2.
    auto *l = root;
    auto *r = root->rchild;
    auto *c = root->rchild->lchild;
    auto *cl = c->lchild;
    auto *cr = c->rchild;
    assert(r->balance == -1);

    // Rotate center and right node to the right.
    r->lchild = cr;
    c->rchild = r;

    // Rotate center and left node to the left.
    l->rchild = cl;
    c->lchild = l;

    // Fix up balance.
    if (c->balance == 0) {
      l->balance = 0;
      r->balance = 0;
    } else if (c->balance > 0) {
      l->balance = -1;
      c->balance = 0;
      r->balance = 0;
    } else {
      l->balance = 0;
      c->balance = 0;
      r->balance = 1;
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
    auto l = root->lchild;
    auto c = root->lchild->rchild;
    auto cl = c->lchild;
    auto cr = c->rchild;
    assert(l->balance == +1);

    // Rotate center and left node to the left.
    l->rchild = cl;
    c->lchild = l;

    // Rotate center and right node to the right.
    r->lchild = cr;
    c->rchild = r;

    // Fix up balance.
    if (c->balance == 0) {
      l->balance = 0;
      r->balance = 0;
    } else if (c->balance > 0) {
      l->balance = -1;
      c->balance = 0;
      r->balance = 0;
    } else {
      l->balance = 0;
      c->balance = 0;
      r->balance = 1;
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
    assert(node->balance == -1);
    if (node->lchild->balance == -1)
      return rotateLR(node);
    return rotateR(node);
  }

  /// The tree rooted by `node` has a temporary balance of +2 (right heavy).
  /// Rebalance the tree by rotating leftwards, and return the new root.
  MemoNode *rebalanceRHeavy(MemoNode *node) {
    assert(node->balance == +1);
    if (node->rchild->balance == -1)
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
      if (child->examinedMaxEnd() > parent->examinedMaxEnd()) {
        parent->examined_max = child->examinedMaxEnd() - parent->sp;
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

      if (slot == &parent->lchild) {
        // Left side grew.

        // Right-heavy => balanced.
        if (parent->balance == +1) {
          parent->balance = 0;
          propMaxExaminedToTop();
          return;
        }

        // Balanced => left-heavy.
        if (parent->balance == 0) {
          parent->balance = -1;
          propMaxExamined(node, parent);
          slot = parentSlot;
          continue;
        }

        // Left-heavy => very-left-heavy.
        if (node->balance == +1)
          *parentSlot = rotateLR(parent);
        else
          *parentSlot = rotateR(parent);
        slot = parentSlot;
      } else {
        // Right side grew.

        // Left-heavy => balanced.
        if (parent->balance < 0) {
          parent->balance = 0;
          propMaxExaminedToTop();
          return;
        }

        // Balanced => right-heavy
        if (parent->balance == 0) {
          parent->balance = +1;
          propMaxExamined(node, parent);
          slot = parentSlot;
          continue;
        }

        // Right-Heavy => Very-Right-Heavy
        if (node->balance < 0)
          *parentSlot = rotateRL(parent);
        else
          *parentSlot = rotateL(parent);
        slot = parentSlot;
      }
    }
  }

  MemoNode *insert(uintptr_t id, const uint8_t *sp, size_t length,
                   size_t examined, std::vector<Capture> &&captures) {
    std::vector<MemoNode **> ancestorSlots;
    auto slot = &root;
    // ancestorSlots.push_back(slot);
    auto node = root;
    while (node) {
      auto order = node->compare(id, sp);
      // if the node is _less_, go to the right slot.
      if (order == Order::LT) {
        ancestorSlots.push_back(slot);
        slot = &node->rchild;
        node = *slot;
        continue;
      }
      // if the node is _more_, go to the left slot.
      if (order == Order::GT) {
        ancestorSlots.push_back(slot);
        slot = &node->lchild;
        node = *slot;
        continue;
      }
      // Otherwise the memo node already exists.
      assert(0);
      return node;
    }

    auto result = new MemoNode(id, sp, length, examined, std::move(captures));
    *slot = result;
    rebalanceInsertion(slot, ancestorSlots);
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Removal
  //////////////////////////////////////////////////////////////////////////////

  /// The tree rooted at `node` has changed height due to a deletion.
  /// `node` is already in AVL shape. Rebalance the ancestors by retracing.
  /// At the point of calling:
  /// - We are rebalancing the parent of node.
  /// - We get the parent by looking at the first element of path.
  /// - The first element of path should be a slot in the grandparent which
  //    contains.
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
      if (slot == &parent->lchild) {
        // Parent: balanced -> right-heavy.
        // Height is unchanged. Stop here.
        if (parent->balance == 0) {
          parent->balance = +1;
          propMaxExaminedToTop();
          return;
        }

        // Parent: left-heavy -> balanced.
        // Height has decreased. Continue retracing the path.
        if (parent->balance == -1) {
          parent->balance = 0;
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
        if (parent->balance == 0) {
          parent->balance = -1;
          propMaxExaminedToTop();
          return;
        }

        // Parent: right-heavy -> balanced.
        // Height has decreased. Continue retracing the path.
        if (parent->balance == +1) {
          parent->balance = 0;
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

  /// Delete the node who is pointed-at by the given slot. path is the path from
  /// the root node down to the target slot. path is used during removal but
  /// will be reset before returning.
  void remove(MemoNode **slot, std::vector<MemoNode **> &path) {
    // Deletion is done by replacing our target with the smallest
    // child on the right. This effectively shrinks the tree on the right,
    // which needs to be rebalanced.
    std::cerr << "!!! remove start\n";
    std::cerr << "path = \n";
    dump(std::cerr, path);
    std::cerr << "slot @" << slot << "=" << *slot << std::endl;
    PathGuard guard(path);
    auto *node = *slot;
    if (node->lchild && node->rchild) {
      // We will replace the deleted node with its successor, which is the least
      // greater node in the tree.
      std::vector<MemoNode **> succPath;
      auto **succSlot = findSuccessor(&node->rchild, succPath);
      auto *succ = *succSlot;
      // Detach the successor node from the tree, and replace it with it's own
      // rchild.
      *succSlot = succ->rchild;
      // Replace the deleted node with the successor, copying over its children.
      node->assignMemoData(std::move(*succ));
      delete succ;
      // Removing the successor could require a rotation.
      rebalanceRemoval(succSlot, succPath);
    } else if (!node->rchild) {
      *slot = node->lchild;
    } else if (!node->lchild) {
      *slot = node->rchild;
    } else {
      *slot = nullptr;
    }

    rebalanceRemoval(slot, path);
  }

  void shift(MemoNode **slot, const uint8_t *sp, signed delta) {
    auto *node = *slot;
    if (node == nullptr)
      return;
    assert(!(node->sp <= sp && sp < node->examinedEnd()) &&
           "node must not intersect insertion point");
    assert(!(node->sp <= sp && sp < node->examinedMaxEnd()) &&
           "node must not intersect insertion point");
    if (node->examinedMaxEnd() <= sp)
      return;
    if (node->sp >= sp)
      sp += delta;
    shift(&node->lchild, sp, delta);
    shift(&node->rchild, sp, delta);
  }

  void invalidate(MemoNode **slot, const uint8_t *low, const uint8_t *high,
                  signed delta, std::vector<MemoNode **> &path) {
    assert(slot && "slot pointer can not be null");
    auto *node = *slot;
    if (!node)
      return;
    std::cerr << "! invalidate begin.n slot @" << slot << "=" << *slot
              << ", low=" << (void *)low << ", high=" << (void *)high
              << std::endl;
    std::cerr << "path = \n";
    dump(std::cerr, path);

    // Left.
    if (low >= node->examinedMaxEnd())
      return;

    path.push_back(slot);
    invalidate(&node->lchild, low, high, delta, path);
    path.pop_back();

    // Center.
    if (node->sp < high && low < node->examinedEnd()) {
      // assert(node->sp < high);
      // assert(node->sp + node->examined >= node->sp);
      remove(slot, path);
    }

    // Right.
    if (high <= node->sp)
      return;

    path.push_back(slot);
    invalidate(&node->rchild, low, high, delta, path);
    path.pop_back();
    std::cerr << "! invalidate end\n";
  }

  void invalidate(const uint8_t *sp, signed inserted, signed removed) {
    if (!inserted && !removed)
      return;
    signed delta = inserted - removed;
    std::vector<MemoNode **> path;
    invalidate(&root, sp, sp + removed, delta, path);
    shift(&root, sp, delta);
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
    dump(os, "L", node->lchild, depth + 1);
    dump(os, "R", node->rchild, depth + 1);
  }

  void dump(std::ostream &os, const MemoNode *node, size_t depth = 0) const {
    dump(os, "C", node, depth);
  }

  void dump(std::ostream &os) const {
    os << "--------\n";
    dump(os, root);
    os << "--------\n";
  }
};

} // namespace hwml
} // namespace circt
#endif