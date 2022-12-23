#include "augur/VM/Augur.h"

//===----------------------------------------------------------------------===//
// Pretty Printing
//===----------------------------------------------------------------------===//

namespace aug {

void print(Object *module);

// struct PPConfig {
//   std::size_t lineLength = 80;
// };

// struct PPState {
//   PPConfig config;
//   PPMode mode;
// };

// enum PPMode {
//   single,
//   broken
// };

// bool fits(Object *object, std::size_t length) {
//   if (auto atom = dyn_cast<Atom>(object))
//     return atom.name.length() <= length;
//   if (auto decl = dyn_cast<Decl>(object))
//     return decl.
//       return false;
// };

// std::ostream pp(PPState &state, Object *object, std::ostream &out) {
//   if (auto atom = dyn_cast<Atom>(object)) {

//     return out;
//   }

//   return out;
// }

// std::ostream pp(Object *object, std::ostream &out) {
//   return pp(PPState, object, out);
// }

// std::ostream pp(Object *obect) { return pp(object, std::cout); }

} // namespace aug