#include "ZZ4b/NtupleAna/interface/myMakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/PythonParameterSet/src/initializeModule.h"

using namespace boost::python;

static
void
makePSetsFromFile(std::string const& fileName, boost::python::object& mainNamespace, int argc, char* argv[]) {
  PySys_SetArgv(argc, argv);//This is the magic line that gives you access to command line arguments in the python config which contains the PSets
  std::string initCommand("from FWCore.ParameterSet.Types import makeCppPSet\n"
                          "execfile('");
  initCommand += fileName + "')";

  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
  std::string command("makeCppPSet(locals(), topPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
}

static
void
makePSetsFromString(std::string const& module, boost::python::object& mainNamespace) {
  std::string command = module;
  command += "\nfrom FWCore.ParameterSet.Types import makeCppPSet\nmakeCppPSet(locals(), topPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
}

namespace edm {
  namespace boost_python {
        
    std::unique_ptr<edm::ParameterSet>
    readPSetsFrom(std::string const& module, int argc, char* argv[]) {
      python::initializeModule();
      
      boost::python::object mainModule = object(handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__")))));
      
      boost::python::object mainNamespace = mainModule.attr("__dict__");
      PythonParameterSet theProcessPSet;
      mainNamespace["topPSet"] = ptr(&theProcessPSet);
      
      try {
	// if it ends with py, it's a file
	if(module.substr(module.size()-3) == ".py") {
	  makePSetsFromFile(module,mainNamespace,argc,argv);
	} else {
	  makePSetsFromString(module,mainNamespace);
	}
      }
      catch( error_already_set const& ) {
	pythonToCppException("Configuration");
	Py_Finalize();
      }
      return std::make_unique<edm::ParameterSet>(ParameterSet(theProcessPSet.pset()));
    }
  } // namespace boost_python
} // namespace edm
