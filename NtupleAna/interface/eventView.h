// -*- C++ -*-

#if !defined(eventView_H)
#define eventView_H
#include <iostream>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/dijet.h"

namespace NtupleAna {
  //eventView object
  class eventView {
    const float lHC = 120;
    const float sHC = 110;
    const float rHC = lHC/sHC;
    const float dHC = 1+pow(rHC, 2);

    float getDHH(float m1, float m2){
      return fabs(m1-m2*rHC)/dHC;
    }

    const float lZC = 90;
    const float sZC = lZC/rHC;

    float getXZZ(float m1, float m2){
      float sigmaLead = (m1-lZC)/(0.1*m1);
      float sigmaSubl = (m2-sZC)/(0.1*m2);
      float xZZ2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xZZ2);
    }

    float getXZH(float m1, float m2){
      float sigmaLead = (m1-lHC)/(0.1*m1);
      float sigmaSubl = (m2-lZC)/(0.1*m2);
      float xZH2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xZH2);
    }

  public:

    dijet* lead;
    dijet* subl;

    dijet* leadSt;
    dijet* sublSt;

    dijet* leadM;
    dijet* sublM;

    TLorentzVector p;
    float pt;
    float eta;
    float phi;
    float m;
    float e;

    float m4j;

    float dHH;

    float xZZ;
    float mZZ;
    bool ZZ;

    float xZH;
    float mZH;
    bool ZH;

    eventView(dijet*, dijet*); 
    ~eventView(); 

    //void dump();
  };

}
#endif // eventView_H

