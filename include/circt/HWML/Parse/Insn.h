#ifndef CIRCT_HWML_PARSE_INSN_H
#define CIRCT_HWML_PARSE_INSN_H

#include <bitset>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace circt {
namespace hwml {

struct Insn;

struct InsnBase {
  enum class Kind {
    Match,
    Jump,
    Choice,
    Call,
    Commit,
    Return,
    Fail,
    End,
    EndFail,
    Set,
    Any,
    PartialCommit,
    BackCommit,
    FailTwice,
    Span,
    CaptureBegin,
    CaptureEnd,
    MemoOpen,
    MemoClose,
    Error
  };
  InsnBase(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

  Kind kind;
};

struct Match : InsnBase {
  Match(std::uint8_t b) : InsnBase(Kind::Match), b(b) {}
  std::uint8_t get() const { return b; }
  std::uint8_t b;
};

struct Jump : InsnBase {
  Jump(const Insn *l) : InsnBase(Kind::Jump), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct Choice : InsnBase {
  Choice(const Insn *l) : InsnBase(Kind::Choice), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct Call : InsnBase {
  Call(const Insn *l) : InsnBase(Kind::Call), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct Commit : InsnBase {
  Commit(const Insn *l) : InsnBase(Kind::Commit), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct Return : InsnBase {
  Return() : InsnBase(Kind::Return) {}
};

struct Fail : InsnBase {
  Fail() : InsnBase(Kind::Fail) {}
};

struct End : InsnBase {
  End() : InsnBase(Kind::End) {}
};

struct EndFail : InsnBase {
  EndFail() : InsnBase(Kind::EndFail) {}
};

struct Set : InsnBase {
  Set(uintptr_t i) : InsnBase(Kind::Set), i(i) {}
  uintptr_t get() const { return i; }
  uintptr_t i;
};

struct Any : InsnBase {
  Any(uintptr_t n) : InsnBase(Kind::Any), n(n) {}
  uintptr_t get() const { return n; }
  uintptr_t n;
};

struct PartialCommit : InsnBase {
  PartialCommit(const Insn *l) : InsnBase(Kind::PartialCommit), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct BackCommit : InsnBase {
  BackCommit(const Insn *l) : InsnBase(Kind::BackCommit), l(l) {}
  const Insn *get() const { return l; }
  const Insn *l;
};

struct FailTwice : InsnBase {
  FailTwice() : InsnBase(Kind::FailTwice) {}
};

struct Span : InsnBase {
  Span(uintptr_t i) : InsnBase(Kind::Span), i(i) {}
  uintptr_t get() const { return i; }
  uintptr_t i;
};

struct CaptureBegin : InsnBase {
  CaptureBegin(uintptr_t id) : InsnBase(Kind::CaptureBegin), id(id) {}
  uintptr_t get() const { return id; }
  uintptr_t id;
};

struct CaptureEnd : InsnBase {
  CaptureEnd() : InsnBase(Kind::CaptureEnd) {}
};

struct MemoOpen : InsnBase {
  MemoOpen(const Insn *l, uintptr_t id)
      : InsnBase(Kind::MemoOpen), l(l), id(id) {}
  const Insn *getLabel() const { return l; }
  uintptr_t getId() const { return id; }
  const Insn *l;
  uintptr_t id;
};

struct MemoClose : InsnBase {
  MemoClose() : InsnBase(Kind::MemoClose) {}
};

struct Error : InsnBase {
  Error(const char *m, const Insn *l) : InsnBase(Kind::Error), m(m), l(l) {}
  std::string getMessage() const { return m; }
  const Insn *getLabel() const { return l; }
  const char *m;
  const Insn *l;
};

struct Insn {
  union {
    InsnBase base;
    Match match;
    Jump jump;
    Choice choice;
    Call call;
    Commit commit;
    Return ret;
    Fail fail;
    End end;
    EndFail endFail;
    Set set;
    Any any;
    PartialCommit partialCommit;
    BackCommit backCommit;
    FailTwice failTwice;
    Span span;
    CaptureBegin captureBegin;
    CaptureEnd captureEnd;
    MemoOpen memoOpen;
    MemoClose memoClose;
    Error error;
  };

  Insn(Match x) : match(x) {}
  Insn(Jump x) : jump(x) {}
  Insn(Choice x) : choice(x) {}
  Insn(Call x) : call(x) {}
  Insn(Commit x) : commit(x) {}
  Insn(Return x) : ret(x) {}
  Insn(Fail x) : fail(x) {}
  Insn(End x) : end(x) {}
  Insn(EndFail x) : endFail(x) {}
  Insn(Set x) : set(x) {}
  Insn(Any x) : any(x) {}
  Insn(PartialCommit x) : partialCommit(x) {}
  Insn(BackCommit x) : backCommit(x) {}
  Insn(FailTwice x) : failTwice(x) {}
  Insn(Span x) : span(x) {}
  Insn(CaptureBegin x) : captureBegin(x) {}
  Insn(CaptureEnd x) : captureEnd(x) {}
  Insn(MemoOpen x) : memoOpen(x) {}
  Insn(MemoClose x) : memoClose(x) {}
  Insn(Error x) : error(x) {}

  InsnBase::Kind getKind() const { return base.getKind(); }
};

struct Program {
  std::vector<Insn> insns;
  std::vector<std::bitset<256>> sets;
};

struct ProgramPrinter {

  ProgramPrinter(const Program &program) : program(program) {
    auto &insns = program.insns;

    std::set<const Insn *> targets;
    auto record = [&](const Insn *insn) { targets.insert(insn); };
    for (auto &insn : insns) {
      switch (insn.getKind()) {
      case InsnBase::Kind::Jump:
        record(insn.jump.l);
        break;
      case InsnBase::Kind::Choice:
        record(insn.choice.l);
        break;
      case InsnBase::Kind::Call:
        record(insn.call.l);
        break;
      case InsnBase::Kind::Commit:
        record(insn.commit.l);
        break;
      case InsnBase::Kind::PartialCommit:
        record(insn.partialCommit.l);
        break;
      case InsnBase::Kind::BackCommit:
        record(insn.backCommit.l);
        break;
      case InsnBase::Kind::MemoOpen:
        record(insn.memoOpen.l);
        break;
      case InsnBase::Kind::Error:
        record(insn.error.l);
        break;
      default:
        break;
      }
    }

    // Record the label number for each target.  They should be iterated in
    // order from the beginning of the program.
    for (auto *insn : targets)
      labels.emplace(insn, labels.size());

    // Calculate how many character wide the maximum label size will be.
    labelWidth = 0;
    if (labels.size())
      labelWidth = log10(labels.size() - 1) + 1;
  }

  void printLabel(std::ostream &os, const Insn *insn) const {
    auto it = labels.find(insn);
    if (it != labels.end())
      os << "L" << std::setfill('0') << std::setw(labelWidth) << it->second;
    else
      os << "<UNKNOWN LABEL>" << (void *)insn;
  };

  void print(std::ostream &os, const Insn &insn) const {
    // Helper to print a label reference inside an instruction.
    // Helper to print a character set.
    auto printSet = [&](uintptr_t index) {
      bool first = true;
      auto &set = program.sets[index];
      unsigned char i = 0;
      do {
        if (set[i]) {
          if (!first)
            os << ", ";
          os << i;
          first = false;
        }
      } while (i++ != 255);
    };

    switch (insn.getKind()) {
    case InsnBase::Kind::Match:
      os << "Match " << insn.match.b;
      break;
    case InsnBase::Kind::Jump:
      os << "Jump ";
      printLabel(os, insn.jump.l);
      break;
    case InsnBase::Kind::Choice:
      os << "Choice ";
      printLabel(os, insn.choice.l);
      break;
    case InsnBase::Kind::Call:
      os << "Call ";
      printLabel(os, insn.call.l);
      break;
    case InsnBase::Kind::Commit:
      os << "Commit ";
      printLabel(os, insn.commit.l);
      break;
    case InsnBase::Kind::Return:
      os << "Return";
      break;
    case InsnBase::Kind::Fail:
      os << "Fail";
      break;
    case InsnBase::Kind::End:
      os << "End";
      break;
    case InsnBase::Kind::EndFail:
      os << "EndFail";
      break;
    case InsnBase::Kind::Set:
      os << "Set ";
      printSet(insn.set.i);
      break;
    case InsnBase::Kind::Any:
      os << "Any " << insn.any.n;
      break;
    case InsnBase::Kind::PartialCommit:
      os << "PartialCommit ";
      printLabel(os, insn.partialCommit.get());
      break;
    case InsnBase::Kind::BackCommit:
      os << "BackCommit ";
      printLabel(os, insn.backCommit.get());
      break;
    case InsnBase::Kind::FailTwice:
      os << "FailTwice";
      break;
    case InsnBase::Kind::Span:
      os << "Span";
      printSet(insn.span.i);
      break;
    case InsnBase::Kind::CaptureBegin:
      os << "CaptureBegin " << insn.captureBegin.get();
      break;
    case InsnBase::Kind::CaptureEnd:
      os << "CaptureEnd";
      break;
    case InsnBase::Kind::MemoOpen:
      os << "MemoOpen ";
      printLabel(os, insn.memoOpen.l);
      os << ", " << insn.memoOpen.id;
      break;
    case InsnBase::Kind::MemoClose:
      os << "MemoClose";
      break;
    case InsnBase::Kind::Error:
      os << "Error \"" << insn.error.m << "\" ";
      printLabel(os, insn.error.l);
      break;
    }
  }

  void print(std::ostream &os) const {
    // Helper to print a label leading a line, or whitespace if there is none.
    auto printLead = [&](const Insn &insn) {
      auto it = labels.find(&insn);
      if (it != labels.end())
        os << "L" << std::setfill('0') << std::setw(labelWidth) << it->second
           << ": ";
      else if (labelWidth)
        os << std::setfill(' ') << std::setw(labelWidth + 3) << "";
    };

    // Print each element.
    for (auto &insn : program.insns) {

      // Print the leading label or whitespace.
      printLead(insn);
      print(os, insn);
      os << std::endl;
    }
  }

private:
  const Program &program;
  std::size_t labelWidth;
  std::unordered_map<const Insn *, size_t> labels;
};

void print(std::ostream &os, const Program &program) {
  ProgramPrinter(program).print(os);
}

///
/// Instruction Stream
///

struct Label {
  explicit Label(uintptr_t index) : index(index) {}
  uintptr_t index;
  Insn *placeholder() const { return (Insn *)index; }
};

struct Index {
  constexpr Index() : index(std::numeric_limits<uintptr_t>::max()) {}
  explicit Index(uintptr_t index) : index(index) {}
  uintptr_t get() const { return index; }
  bool operator==(const Index &rhs) const { return get() == rhs.get(); }

private:
  uintptr_t index;
};

constexpr Index INVALID_INDEX;

struct InsnStream {

private:
  template <typename T, typename... Args>
  Index push(Args &&...args) {
    auto size = insns.size();
    insns.emplace_back(T(std::forward<Args>(args)...));
    return Index(size);
  };

  uintptr_t getSetIndex(std::bitset<256> set) {
    auto [iter, inserted] = setMap.emplace(set, sets.size());
    if (inserted)
      sets.push_back(set);
    return iter->second;
  }

public:
  Label label() {
    auto size = labels.size();
    labels.emplace_back();
    return Label(size);
  }

  Label label(Index index) {
    auto l = label();
    setLabel(l, index);
    return l;
  }

  void setLabel(Label label, Index index) {
    assert(labels[label.index] == INVALID_INDEX);
    labels[label.index] = index;
  }

  Index match(uint8_t b) { return push<Match>(b); }
  Index jump(Label l) { return push<Jump>(l.placeholder()); }
  Index choice(Label l) { return push<Choice>(l.placeholder()); }
  Index call(Label l) { return push<Call>(l.placeholder()); }
  Index commit(Label l) { return push<Commit>(l.placeholder()); }
  Index ret() { return push<Return>(); }
  Index fail() { return push<Fail>(); }
  Index end() { return push<End>(); }
  Index endFail() { return push<EndFail>(); }
  Index set(std::bitset<256> set) { return push<Set>(getSetIndex(set)); }
  Index set(std::string str) {
    std::bitset<256> bitset;
    for (auto c : str)
      bitset.set(c);
    return set(bitset);
  }
  Index any(uintptr_t n) { return push<Any>(n); }
  Index partialCommit(Label l) { return push<PartialCommit>(l.placeholder()); }
  Index backCommit(Label l) { return push<BackCommit>(l.placeholder()); }
  Index failTwice() { return push<FailTwice>(); }
  Index span(std::bitset<256> set) { return push<Span>(getSetIndex(set)); }
  Index captureBegin(uintptr_t id) { return push<CaptureBegin>(id); }
  Index captureEnd() { return push<CaptureEnd>(); }
  Index memoOpen(Label l, uintptr_t id) {
    return push<MemoOpen>(l.placeholder(), id);
  }
  Index memoClose() { return push<MemoClose>(); }
  Index error(const char *m, Label l) {
    return push<Error>(m, l.placeholder());
  }

  Program finalize() {
    // All instruction pointers need to be finalized
    auto fixup = [&](const Insn *&insn) {
      insn = insns.data() + labels[(std::size_t)insn].get();
    };
    for (auto &insn : insns) {
      switch (insn.getKind()) {
      case InsnBase::Kind::Jump:
        fixup(insn.jump.l);
        break;
      case InsnBase::Kind::Choice:
        fixup(insn.choice.l);
        break;
      case InsnBase::Kind::Call:
        fixup(insn.call.l);
        break;
      case InsnBase::Kind::Commit:
        fixup(insn.commit.l);
        break;
      case InsnBase::Kind::PartialCommit:
        fixup(insn.partialCommit.l);
        break;
      case InsnBase::Kind::BackCommit:
        fixup(insn.backCommit.l);
        break;
      case InsnBase::Kind::MemoOpen:
        fixup(insn.memoOpen.l);
        break;
      case InsnBase::Kind::Error:
        fixup(insn.error.l);
        break;
      default:
        break;
      }
    }

    return {std::move(insns), std::move(sets)};
  }

private:
  std::vector<Insn> insns;
  std::vector<std::bitset<256>> sets;
  std::unordered_map<std::bitset<256>, uintptr_t> setMap;
  std::vector<Index> labels;
};

} // namespace hwml
} // namespace circt

#endif