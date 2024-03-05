#ifndef CIRCT_HWML_PARSE_CAPTURE_H
#define CIRCT_HWML_PARSE_CAPTURE_H

#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

namespace circt {
namespace hwml {

struct Capture {
  Capture(std::uintptr_t id, const std::uint8_t *start, const std::uint8_t *end,
          std::vector<Capture> &&children)
      : id(id), start(start), end(end), children(std::move(children)) {}
  // Capture(Capture &&other) = default;
  std::uintptr_t id;
  const std::uint8_t *start;
  const std::uint8_t *end;
  std::vector<Capture> children;
};

void print(std::ostream &os, std::size_t indent, const Capture &capture);
void print(std::ostream &os, const Capture &capture);
void print(std::ostream &os, std::size_t indent,
           const std::vector<Capture> &captures);
void print(std::ostream &os, const std::vector<Capture> &captures);

} // namespace hwml
} // namespace circt

#endif // CIRCT_HWML_PARSE_CAPTURE_H
