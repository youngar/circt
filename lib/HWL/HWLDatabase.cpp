#include "circt/HWL/HWLDatabase.h"
#include "circt/HWL/HWLParser.h"

using namespace circt;
using namespace circt::hwl;

void HWLDatabase::addDocument(StringRef filename, const std::string &contents) {
    files.try_emplace(filename, 
}

inc::Input<HWLDocument> &HWLDatabase::getDocument(StringRef filename) {
  return *files.at(filename);
}

const inc::Input<HWLDocument> &
HWLDatabase::getDocument(StringRef filename) const {
  return *files.at(filename);
}