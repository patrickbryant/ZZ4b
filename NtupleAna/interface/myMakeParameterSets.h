// -*- C++ -*-

#if !defined(myMakeParameterSets_H)
#define myMakeParameterSets_H

//----------------------------------------------------------------------
// Declare functions used to create ParameterSets.
//----------------------------------------------------------------------

#include <memory>

#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  namespace  boost_python {
        
    /**finds all the PSets used in the top level module referred as a file or as a string containing
       python commands. These PSets are bundled into a top level PSet from which they can be retrieved.
       If the PSets are in a python config file, command line arguments can be parsed inside the python config file.
    */
    std::unique_ptr<edm::ParameterSet> readPSetsFrom(std::string const& fileOrString, int argc, char* argv[]);
  } // BoostPython
} // namespace edm
#endif
