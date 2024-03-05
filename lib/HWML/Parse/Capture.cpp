#include "circt/HWML/Parse/Capture.h"
#include <iomanip>

using namespace circt;
using namespace hwml;

void circt::hwml::print(std::ostream &os, std::size_t indent,
                        const Capture &capture) {
  os << std::setfill(' ') << std::setw(indent) << "";
  os << capture.id << ": \""
     << std::string_view{(const char *)capture.start,
                         static_cast<std::size_t>(capture.end - capture.start)}
     << "\"";
  for (auto &c : capture.children) {
    os << std::endl;
    print(os, indent + 2, c);
  }
}

void circt::hwml::print(std::ostream &os, const Capture &capture) {
  print(os, 0, capture);
}

void circt::hwml::print(std::ostream &os, std::size_t indent,
                        const std::vector<Capture> &captures) {
  bool first = true;
  for (const auto &capture : captures) {
    if (!first)
      os << std::endl;
    first = false;
    print(os, indent, capture);
  }
}

void circt::hwml::print(std::ostream &os,
                        const std::vector<Capture> &captures) {
  print(os, 0, captures);
}
