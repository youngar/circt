#include "circt/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include <stdint.h>

#ifndef CIRCT_HWML_HWMLINCREMENTAL_H
#define CIRCT_HWML_HWMLINCREMENTAL_H
namespace circt {

namespace hwml {

class Database {
public:
  template <typename T, typename... Args>
  void registerQuery(Args &...args) {}

  template <typename Input>
  void addInput(Input input) {}

  template <typename InputT, typename... Args>
  void updateInput(typename InputT::Params params, Args... args) {}

  template <typename Query, typename... Args>
  Query query(Args... args) {}

private:
  Revision revision;

  DenseMap<TypeID, void *> queryMap;
};

} // namespace hwml
} // namespace circt

#endif
