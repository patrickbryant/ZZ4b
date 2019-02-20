#include "ZZ4b/NtupleAna/interface/myParameterSetReader.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "ZZ4b/NtupleAna/interface/myMakeParameterSets.h"

std::unique_ptr<edm::ParameterSet> edm::readPSetsFrom(std::string const& fileOrString, int argc, char* argv[]) {
  return edm::boost_python::readPSetsFrom(fileOrString, argc, argv);
}
