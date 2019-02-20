// -*- C++ -*-

#if !defined(myParameterSetReader_H)
#define myParameterSetReader_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  std::unique_ptr<edm::ParameterSet> readPSetsFrom(std::string const& fileOrString, int argc, char* argv[]);


};
#endif
