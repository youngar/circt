#ifndef CIRCT_HWML_PARSE_MACHINE_H
#define CIRCT_HWML_PARSE_MACHINE_H

#include "circt/HWML/Parse/Capture.h"
#include "circt/HWML/Parse/Insn.h"
#include "circt/HWML/Parse/MemoTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdint.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace circt {
namespace hwml {

///
/// Machine
///

struct Entry {

  enum class Kind { Ret, Backtrack, Capture, Memo };

private:
  Entry(Kind kind, const Insn *ip, const uint8_t *sp, uintptr_t id)
      : kind(kind), ip(ip), sp(sp), id(id) {}

public:
  /// Create a return stack frame.
  static Entry ret(const Insn *ip) {
    assert(ip && "instruction pointer cannot be null");
    return Entry(Kind::Ret, ip, nullptr, invalidId);
  }

  /// Create a backtracking stack entry.
  static Entry backtrack(const Insn *ip, const uint8_t *sp) {
    std::cerr << "#### ip=" << (void *)ip << ", sp=" << (void *)sp << std::endl;
    assert(ip && "instruction pointer cannot be null");
    assert(sp && "subject pointer cannot be null");
    return Entry(Kind::Backtrack, ip, sp, invalidId);
  }

  /// Create a capture stack entry.
  static Entry capture(uintptr_t id, const uint8_t *begin) {
    assert(id != invalidId && "can't use an invalid id");
    assert(begin && "begin pointer cannot be null");
    return Entry(Kind::Capture, nullptr, begin, id);
  }

  /// Create a memo entry.
  static Entry memo(const uint8_t *sp, uintptr_t id) {
    assert(sp && "subject pointer cannot be null");
    assert(id != invalidId && "can't use an invalid id");
    return Entry(Kind::Memo, nullptr, sp, id);
  }

  bool isReturn() const { return kind == Kind::Ret; }
  bool isBacktrack() const { return kind == Kind::Backtrack; }
  bool isCapture() const { return kind == Kind::Capture; }
  bool isMemo() const { return kind == Kind::Memo; }

  const Insn *getIP() const { return ip; }
  void setIP(const Insn *ip) { this->ip = ip; }

  const uint8_t *getSP() const { return sp; }
  void setSP(const uint8_t *sp) { this->sp = sp; }

  uintptr_t getID() const { return id; }

  template <typename... Args>
  Capture &capture(Args &&...args) {
    return captures.emplace_back(args...);
  }

  /// The list of children captures.
  std::vector<Capture> captures;

private:
  static constexpr uintptr_t invalidId = std::numeric_limits<uintptr_t>::max();

  /// The kind of the strack frame.
  Kind kind;

  /// The instruction pointer to return to.
  const Insn *ip;

  /// If this is a backtrack frame, this will contain the subject pointer to
  /// return to. Otherwise, it will contain nullptr.
  const uint8_t *sp;

  /// If this is a capture frame, then this will contain the id of the
  /// capture. Otherwise it will be set to -1.
  uintptr_t id;
};

struct Diagnostic {
  Diagnostic(const std::string &message, const uint8_t *sp)
      : message(message), sp(sp) {}
  std::string message;
  const uint8_t *sp;
};

struct Machine {

  Machine(const Program &program, MemoTable &memoTable, const uint8_t *sp,
          const uint8_t *se)
      : program(program), ip(program.insns.data()), bp(sp), sp(sp), se(se),
        ep(sp), sets(program.sets.data()), memoTable(memoTable) {}

  void dumpStack(const ProgramPrinter &printer) {
    std::cerr << "stack dump\n";
    for (std::size_t i = 0, e = stack.size(); i < e; ++i) {
      std::cerr << i << ": ";
      auto &entry = stack[e - i - 1];
      if (entry.isBacktrack()) {
        std::cerr << "backtrack ip=";
        printer.printLabel(std::cerr, entry.getIP());
        std::cerr << ", sp=" << (void *)entry.getSP();
      } else if (entry.isCapture()) {
        std::cerr << "capture id=" << entry.getID()
                  << ", begin=" << (void *)entry.getSP();
      } else if (entry.isReturn()) {
        std::cerr << "return ip=" << (void *)entry.getIP();
      } else {
        std::cerr << "memo id" << (void *)entry.getID()
                  << ", sp=" << (void *)entry.getSP();
      }
      if (!entry.captures.empty()) {
        std::cerr << std::endl;
        print(std::cerr, 4, entry.captures);
      }
      std::cerr << std::endl;
    }
    if (!captures.empty()) {
      std::cerr << std::endl << "top level captures" << std::endl;
      print(std::cerr, 4, captures);
    }
    std::cerr << std::endl;
  };

  Position toPosition(const uint8_t *p) {
    assert(bp <= p && p <= ep && "pointer outside of the buffer");
    return (Position)(p - bp);
  }

  /// Pops the top entry from the stack and returns it. Discards any current
  /// captures.
  Entry popEntry() {
    assert(!stack.empty() && "stack cannot be empty");
    auto entry = std::move(stack.back());
    stack.pop_back();
    return entry;
  }

  /// Propogate the captures in the current stack frame to the parent frame's
  /// captures.  If there is no parent frame, propagate the captures to the
  /// global capture list.
  void propagateEntry() {
    assert(stack.size() && "stack cannot be empty");
    std::cerr << "!!propagateEntry:\n";
    auto moveCaptures = [](auto &from, auto &to) {
      std::move(from.begin(), from.end(), std::back_inserter(to));
      from.clear();
    };

    if (stack.size() > 1) {
      // If there is a parent stack element, append the current captures to
      // it.
      moveCaptures(stack.back().captures, stack[stack.size() - 2].captures);
    } else {
      // If the stack is empty, add these captures to the global list of
      // captures.
      moveCaptures(stack.back().captures, captures);
    }
  }

  /// Pops the top entry from the stack, collecting the curren captures in to
  /// the parent frame.
  Entry propAndPop() {
    propagateEntry();
    return popEntry();
  }

  void memoize(uintptr_t id, const uint8_t *sp, size_t spOff, size_t epOff,
               const std::vector<Capture> &captures) {
    memoTable.insert(id, toPosition(sp), spOff, epOff,
                     std::vector<Capture>(captures));
  }

  void memoizeSuccess(uintptr_t id, const uint8_t *sp, const uint8_t *spEnd,
                      const uint8_t *epEnd,
                      const std::vector<Capture> &captures) {
    size_t spOff = spEnd - sp;
    size_t epOff = epEnd - sp;
    memoize(id, sp, spOff, epOff, captures);
  }

  void memoizeFailure(uintptr_t id, const uint8_t *sp, const uint8_t *epEnd) {
    llvm::errs() << "!!!! memoizeFailure id=" << id << ", sp=" << sp
                 << ", epEnd=" << epEnd << "\n";
    // Ensure the length is -1.
    size_t spOff = -1;
    // Number of examined bytes needs to be accurate.
    size_t epOff = epEnd - sp;
    assert(epEnd >= sp);
    memoize(id, sp, spOff, epOff, {});
  }

  /// Advances ip and comsumes one byte from the subject if it matches b, and
  /// goes to the failstate otherwise.
  bool match(uint8_t b) {
    // Check if we are at the end of the subject.
    if (sp == se)
      return fail();


    // Check if we match the current byte.
    if (*sp != b)
      return fail();

    // Increment the instruction and subject pointer.
    ++ip;
    ++sp;
    ep = std::max(sp, ep);
    return false;
  }

  /// Sets ip to l.
  void jump(const Insn *l) { ip = l; }

  /// Pushes a backtrack entry storing l and sp so that the parser can return
  /// to this position in the document later and parse a different pattern.
  void choice(const Insn *l) {
    stack.emplace_back(Entry::backtrack(l, sp));
    ++ip;
  }

  /// Pushes the next ip to the stack as a return address and jumps to l.
  /// Calls will be used to implement non-terminals.
  void call(const Insn *l) {
    stack.emplace_back(Entry::ret(ip + 1));
    ip = l;
  }

  /// Pops the top entry off the stack and jumps to l.  This allows the
  /// machine to commit to a state and discard a backtrack entry.
  void commit(const Insn *l) {
    assert(stack.back().isBacktrack() && "can't commit a non-backtrack entry");
    propAndPop();
    ip = l;
  }

  /// Pops a return address from the stack and jumps to it.
  void ret() {
    auto entry = propAndPop();
    assert(entry.isReturn() && "can't return to a non-return entry");
    ip = entry.getIP();
  }

  /// Pops the stack until we find a backtrack entry.  If one is found,
  /// continue parsing as normal. If no backtrack is found then fail to match
  /// the subject.
  bool fail() {
    ProgramPrinter printer(program);
    while (!stack.empty()) {
      auto entry = stack.back();
      stack.pop_back();
      if (entry.isBacktrack()) {
        ip = entry.getIP();
        sp = entry.getSP();
        return false;
      }

      if (entry.isMemo()) {
        memoizeFailure(entry.getID(), sp, ep);
      }
    }
    return true;
  }

  /// Advances ip and consumes one byte from the subject if it is contained by
  /// the character set X, and goes to the fail state otherwise.
  bool set(uintptr_t i) {
    if (sp == se)
      return fail();

    if (sets[i][*sp]) {
      ++sp;
      ep = std::max(sp, ep);
      ++ip;
      return false;
    }

    return fail();
  }

  /// Match n characters.
  bool any(uintptr_t n) {
    if ((uintptr_t)(se - sp) < n) {
      sp = se;
      return fail();
    }
    sp += n;
    ep = std::max(sp, ep);
    ++ip;
    return false;
  }

  /// Update the backtrack entry on the top of the stack to the current
  /// subject position and jump to l.  This effectively performs a commit and
  /// choice in one instruction.
  void partialCommit(const Insn *l) {
    auto &entry = stack.back();
    assert(entry.isBacktrack() && "can't commit a non-backtrack entry");
    propagateEntry();
    entry.setSP(sp);
    ip = l;
  }

  /// Pops the backtrack entry on the top backtrack of the stack and updates
  /// the sp to point to the entry's sp, and then jumps to l.
  void backCommit(const Insn *l) {
    auto entry = stack.back();
    assert(entry.isBacktrack() && "can't commit a non-backtrack entry");
    sp = entry.getSP();
    ip = l;
  }

  /// Pops the top entry off the stack and then fails.
  bool failTwice() {
    assert(!stack.empty() && "requires at least one stack entry");
    stack.pop_back();
    return fail();
  }

  /// Consumes input and advances sp as long as the input matches the
  /// character set.  This instruction never fails, but might not consume any
  /// input if there is no match.
  void span(uintptr_t i) {
    while (sp == se) {
      if (!sets[i][*sp])
        return;
      ++sp;
      ep = std::max(sp, ep);
      ++ip;
    }
  }

  /// Start a capture by pushing a capture frame with the specified id.
  void captureBegin(uintptr_t id) {
    stack.emplace_back(Entry::capture(id, sp));
    ++ip;
  }

  /// Pops a capture entry off of the stack, and creates a new capture object
  /// which is appended to the captures of the new stack top.
  void captureEnd() {
    auto entry = popEntry();
    assert(entry.isCapture() && "must be a capture frame");
    if (!stack.empty()) {
      // Add to the parent frame's captures.
      stack.back().captures.emplace_back(entry.getID(), entry.getSP(), sp,
                                         std::move(entry.captures));
    } else {
      // Add to the global list of captures.
      captures.emplace_back(entry.getID(), entry.getSP(), sp,
                            std::move(entry.captures));
    }
    ++ip;
  }

  /// If there is a memoization entry corresponding to (id, sp), jump to l,
  /// and advance the sp by the length of the memoization. If there is no
  /// corresponding memoization entry, a memoization stack entry is pushed.
  bool memoOpen(const Insn *l, uintptr_t id) {
    auto *node = memoTable.lookup(id, toPosition(sp));
    if (node) {
      if (node->getLength() == std::numeric_limits<std::size_t>::max()) {
        return fail();
      }
      ip = l;
      sp += node->getLength();
      ep = std::max(sp, ep);
      stack.back().captures.insert(stack.back().captures.end(),
                                   node->getCaptures().begin(),
                                   node->getCaptures().end());
      return false;
    }
    stack.emplace_back(Entry::memo(sp, id));
    ++ip;
    return false;
  }

  void memoClose() {
    auto &entry = stack.back();
    assert(entry.isMemo() && "must be memoframe");
    memoizeSuccess(entry.getID(), entry.getSP(), sp, ep, entry.captures);
    // Return the captures.
    propAndPop();
    ++ip;
  }

  void error(const std::string &m, const Insn *l) {
    diagnostics.emplace_back(m, sp);
    ip = l;
  }

  bool run() {

    ProgramPrinter printer(program);

    while (true) {
#if 0
      dumpStack(printer);
      std::cerr << "Executing ";
      printer.print(std::cerr, *ip);
      std::cerr << std::endl;
#endif
      switch (ip->getKind()) {
      case InsnBase::Kind::Match:
        if (match(ip->match.get()))
          return false;
        break;
      case InsnBase::Kind::Jump:
        jump(ip->jump.get());
        break;
      case InsnBase::Kind::Choice:
        choice(ip->choice.get());
        break;
      case InsnBase::Kind::Call:
        call(ip->call.get());
        break;
      case InsnBase::Kind::Commit:
        commit(ip->commit.get());
        break;
      case InsnBase::Kind::Return:
        ret();
        break;
      case InsnBase::Kind::Fail:
        if (fail())
          return false;
        break;
      case InsnBase::Kind::End:
        return true;
        break;
      case InsnBase::Kind::EndFail:
        return false;
        break;
      case InsnBase::Kind::Set:
        if (set(ip->set.get()))
          return false;
        break;
      case InsnBase::Kind::Any:
        if (any(ip->any.get()))
          return false;
        break;
      case InsnBase::Kind::PartialCommit:
        partialCommit(ip->partialCommit.get());
        break;
      case InsnBase::Kind::BackCommit:
        backCommit(ip->backCommit.get());
        break;
      case InsnBase::Kind::FailTwice:
        if (failTwice())
          return false;
        break;
      case InsnBase::Kind::Span:
        span(ip->span.get());
        break;
      case InsnBase::Kind::CaptureBegin:
        captureBegin(ip->captureBegin.get());
        break;
      case InsnBase::Kind::CaptureEnd:
        captureEnd();
        break;
      case InsnBase::Kind::MemoOpen:
        if (memoOpen(ip->memoOpen.l, ip->memoOpen.id))
          return false;
        break;
      case InsnBase::Kind::MemoClose:
        memoClose();
        break;
      case InsnBase::Kind::Error:
        error(ip->error.m, ip->error.l);
        break;
      }
    }
    return false;
  }

  static bool parse(const Program &program, MemoTable &memoTable,
                    const uint8_t *sp, const uint8_t *se,
                    std::vector<Capture> &captures,
                    std::vector<Diagnostic> &diagnostics) {
    Machine machine(program, memoTable, sp, se);
    auto result = machine.run();
    captures = std::move(machine.captures);
    diagnostics = std::move(machine.diagnostics);
    machine.captures.clear();
    return result;
  }

  static bool parse(const Program &program, const uint8_t *sp,
                    const uint8_t *se, std::vector<Capture> &captures,
                    std::vector<Diagnostic> &diagnostics) {
    MemoTable memoTable;
    return Machine::parse(program, memoTable, sp, se, captures, diagnostics);
  }

  /// The program.
  const Program &program;
  /// Instruction pointer.
  const Insn *ip;
  /// Beginning pointer.
  const uint8_t *bp;
  /// Subject pointer.
  const uint8_t *sp;
  /// Subject end.
  const uint8_t *se;
  /// The furthes subject pointer examined so far.
  const uint8_t *ep;
  /// The current line of input.
  uintptr_t line = 0;
  /// An array of bitsets.
  const std::bitset<256> *sets;
  /// The run time activation frame stack.
  std::vector<Entry> stack;
  /// The top level captures.
  std::vector<Capture> captures;
  MemoTable &memoTable;
  std::vector<Diagnostic> diagnostics;
};

} // namespace hwml
} // namespace circt

#endif