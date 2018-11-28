// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

namespace NtupleAna {

  class eventData {

  public:
    TChain* tree;
    bool debug;
    UInt_t    run     =  0;
    ULong64_t event   =  0;
    float     weight  =  1;

  private:
    UInt_t    run_arr         [1] = {0};
    ULong64_t event_arr       [1] = {0};
    float     genWeight_arr   [1] = {0};

  public:

    eventData(TChain*, bool); 

    ~eventData(); 

    void update(int);

    void dump();
  };

}
#endif // eventData_H
