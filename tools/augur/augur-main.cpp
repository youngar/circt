#include "augur/Parser/AugurParser.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace aug;
using namespace mlir;

struct File {
  LogicalResult open(const std::string &filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in)
      return failure();
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return success();
  }

  std::string_view getContents() { return contents; }

private:
  std::string contents;
};

int main(int argc, char *argv[]) {
  std::cout << "Starting\n";

  File file;
  if (argc > 1) {
    std::string filename(argv[1]);
    std::cout << "opening file: " << filename << std::endl;
    if (failed(file.open(filename))) {
      std::cerr << "failed to open the file";
      return EXIT_FAILURE;
    }

    std::cout << "contents: " << file.getContents() << std::endl;
  }

  VirtualMachine vm;
  if (!vm.initialize())
    return EXIT_FAILURE;

  auto *module = parse(vm, file.getContents());
  if (!module)
    return EXIT_FAILURE;

  std::cout << "Parsed this module:\n" << *module << "\n";

  auto result = aug::typeCheck(vm, module);
  if (succeeded(result)) {
    std::cout << "succeeded to type check\n";
  } else {
    std::cout << "failure to type check\n";
  }

  std::cout << "Evaluating\n" << *aug::eval(vm, module) << "\n";

  return EXIT_SUCCESS;
}
