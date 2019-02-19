// -*- C++ -*-

#if !defined(eventView_H)
#define eventView_H
#include <iostream>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/dijet.h"

namespace NtupleAna {
  //eventView object
  class eventView {

    //DiJet Mass Plane Region Definitions
    const float leadHH = 120.0; const float sublHH = 110.0;
    const float leadZH = 120.0; const float sublZH =  90.0;
    const float leadZZ =  82.5; const float sublZZ =  90.0;

    const float xZZSR =  1.60;
    const float rZZCR = 28.00;
    const float sZZCR =  1.02;
    const float rZZSB = 40.00;
    const float sZZSB =  1.04;

    const float xZHSR =  1.60;
    const float rZHCR = 30.00;
    const float sZHCR =  1.03;
    const float rZHSB = 45.00;
    const float sZHSB =  1.05;

    const float xHHSR =  1.60;
    const float rHHCR = 30.00;
    const float sHHCR =  1.03;
    const float rHHSB = 45.00;
    const float sHHSB =  1.05;

    const float slopeDBB = leadHH/sublHH;
    const float denomDBB = sqrt(1+pow(slopeDBB, 2));

    float getDBB(float m1, float m2){
      return fabs(m1-m2*slopeDBB)/denomDBB;
    }

    float getXZZ(float m1, float m2){
      float sigmaLead = (m1-leadZZ)/(0.1*m1);
      float sigmaSubl = (m2-sublZZ)/(0.1*m2);
      float xZZ2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xZZ2);
    }

    float getXZH(float m1, float m2){
      float sigmaLead = (m1-leadZH)/(0.1*m1);
      float sigmaSubl = (m2-sublZH)/(0.1*m2);
      float xZH2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xZH2);
    }

    float getXHH(float m1, float m2){
      float sigmaLead = (m1-leadHH)/(0.1*m1);
      float sigmaSubl = (m2-sublHH)/(0.1*m2);
      float xHH2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xHH2);
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

    float dBB;
    float xZZ;
    float xZH;
    float xHH;

    float mZZ;
    float mZH;
    float mHH;

    bool ZZSR;
    bool ZHSR;
    bool HHSR;

    eventView(dijet*, dijet*); 
    ~eventView(); 

    //void dump();
  };

}
#endif // eventView_H

