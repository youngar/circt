#ifndef CIRCT_HWL_PARSE_LINEINFOTABLE_H
#define CIRCT_HWL_PARSE_LINEINFOTABLE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <vector>

namespace circt {
namespace hwl {

/// A table mapping file offsets into lines and column positions.
///
class LineInfoTable {
public:
  LineInfoTable() { update(""); }

  explicit LineInfoTable(StringRef content) : LineInfoTable() {
    update(content);
  }

  void update(StringRef content) {
    breaks.clear();
    // The start of the first line is implicit.
    store(0);
    for (auto i = content.begin(), e = content.end(); i != e; ++i) {
      if (*i == '\n')
        store(std::distance(content.begin(), i));
    }
  }

  void update(std::size_t line, std::size_t column, StringRef content) {}

  void update(std::size_t offset, StringRef content, std::size_t removed) {
    // Create a list of all the new lines inserted.
    SmallVector<std::size_t> inserted;
    for (auto i = content.begin(), e = content.end(); i != e; ++i) {
      if (*i == '\n')
        inserted.push_back(offset + std::distance(content.begin(), i));
    }

    auto line = getLineForOffset(offset);
    auto insertedAmt = inserted.size();
    auto removedAmt = line - getLineForOffset(offset + removed);
    // ssize_t lineDelta = insertedAmt - removedAmt;
    auto contentDelta = content.size() - removed;

    if (insertedAmt >= removedAmt) {
      // If we're inserting more than we removed, we overwrite as many removed
      // as we can, and then insert the remainder.
      for (auto i = 0ul; i < removedAmt; ++i)
        breaks[i + line] = inserted[i];

      // Insert any extra elements.
      breaks.insert(breaks.begin() + line + removedAmt,
                    inserted.begin() + removedAmt, inserted.end());

    } else {
      // If we're removing more than we are inserting, we can copy in the new
      // line information, and then shift down the remaining elements.
      auto delta = removedAmt - insertedAmt;
      for (auto i = 0ul; i < insertedAmt; ++i)
        breaks[i + line] = inserted[i];

      // Remove any lost elements.
      breaks.erase(breaks.begin() + line + insertedAmt,
                   breaks.begin() + line + insertedAmt + delta);
    }

    // Update the offsets of any trailing elements.
    for (auto i = breaks.begin() + line + insertedAmt, e = breaks.end(); i != e;
         ++i)
      *i += contentDelta;
  }

  /// Get the start offset of a line.
  std::size_t getOffsetForLine(std::size_t n) const { return breaks[n]; }

  std::size_t getOffsetForLineCol(std::size_t line, std::size_t col) const {
    return getOffsetForLine(line) + col;
  }

  /// Get the end offset of a line.
  std::size_t getOffsetForLineEnd(std::size_t n) const { return breaks[n + 1]; }

  /// True if offset is within line number n.
  bool isOffsetInLine(std::size_t offset, std::size_t n) const {
    return getOffsetForLine(n) <= offset && offset < getOffsetForLineEnd(n);
  }

  /// Find the line number of a position. Lines are 0-indexed.
  std::size_t getLineForOffset(std::size_t offset) const {
    const auto n = breaks.size();

    // If the offset is _after_ the last line break, the offset occurs on
    // the last line. The last line has no upper-limit, so we have to treat
    // it specially.
    if (breaks[n - 1] <= offset) {
      return n - 1;
    }

    // Otherwise, search for the line who's interval contains the offset.
    std::size_t l = 0;
    std::size_t r = n - 1;
    while (l <= r) {
      std::size_t m = (l + r) / 2;
      if (getOffsetForLineEnd(m) < offset + 1) {
        l = m + 1;
      } else if (offset < getOffsetForLine(m)) {
        r = m - 1;
      } else {
        return m;
      }
    }
    llvm_unreachable("should not reach here");
    return 0;
  }

  std::pair<std::size_t, std::size_t>
  getLineAndColForOffset(std::size_t offset) const {
    auto line = getLineForOffset(offset);
    auto col = offset - getOffsetForLine(line);
    return {line, col};
  }

  /// Find the column number of a position. Columns are 0-indexed.
  std::size_t getColumnForOffset(std::size_t o) const {
    return o - getOffsetForLine(getLineForOffset(o));
  };

  std::pair<std::size_t, std::size_t>
  getLineAndColumnForOffset(std::size_t o) const {
    auto line = getLineForOffset(o);
    auto column = o - getOffsetForLine(line);
    return {line, column};
  }

private:
  /// Note a linebreak at position p.
  void store(std::size_t offset) { breaks.push_back(offset); }

  std::vector<std::size_t> breaks;
};

} // namespace hwl
} // namespace circt

#endif