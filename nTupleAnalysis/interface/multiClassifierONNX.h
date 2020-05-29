// -*- C++ -*-
#if !defined(multiClassifierONNX_H)
#define multiClassifierONNX_H

#include <iostream>
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
//#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

//using namespace cms::Ort;

namespace nTupleAnalysis {
  class eventData;

  class multiClassifierONNX {
  public:

    multiClassifierONNX(std::string modelFile);
    
    std::unique_ptr<cms::Ort::ONNXRuntime> model;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    cms::Ort::FloatArrays input;
    cms::Ort::FloatArrays output;
    std::vector<float> c_score;
    std::vector<float> q_score;

    const std::vector<int> canJetImageIndicies = {0,1,2,3, 0,2,1,3, 0,3,1,2};

    void clear();
    void loadInput(eventData* event);
    void run();
    void run(eventData* event);
    void dump();
    
    ~multiClassifierONNX(); 

  };

}
#endif // multiClassifierONNX_H
