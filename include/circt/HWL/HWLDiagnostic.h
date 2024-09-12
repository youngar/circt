#ifndef CIRCT_HWL_HWLDIAGNOSTIC_H
#define CIRCT_HWL_HWLDIAGNOSTIC_H

#include "circt/HWL/HWLLocation.h"
#include <vector>

namespace circt {
namespace hwl {

/// A diagnostic message produced by the parser.
struct Diagnostic {
  Diagnostic(const std::string &message, Location location)
      : message(message), location(location) {}

  /// Get the message of this diagnostic.
  std::string getMessage() const { return message; }

  /// Get this location of this diagnostic.
  Location getLocation() const { return location; }

  /// Set this location of this diagnostic.
  void setLocation(Location location) { this->location = location; }

private:
  std::string message;
  Location location;
};

using DiagnosticManager = std::vector<Diagnostic>;

} // namespace hwl
} // namespace circt
#endif