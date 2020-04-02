// -*- C++ -*-
#if !defined(multiClassifierONNX_H)
#define multiClassifierONNX_H

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace nTupleAnalysis {

  class multiClassifierONNX {
  public:

    multiClassifierONNX(std::string modelFile);

    
    ~multiClassifierONNX(); 

  };

}
#endif // multiClassifierONNX_H
