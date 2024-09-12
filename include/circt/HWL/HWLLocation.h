#ifndef CIRCT_HWL_HWLLOCATION_H
#define CIRCT_HWL_HWLLOCATION_H

#include <cstddef>
#include <string>

/// A location in a source file.
struct Location {
  /// The name of the file.
  std::string filename;
  /// The current column.
  std::size_t column;
  /// The current line.
  std::size_t line;
};

#endif // CIRCT_HWL_HWLLOCATION_H