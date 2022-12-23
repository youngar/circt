
#include "augur/VM/Augur.h"
#include "mlir/Support/LogicalResult.h"
#include <string_view>

namespace aug {
Module *parse(VirtualMachine &vm, std::string_view code);
} // namespace aug
