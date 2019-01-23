// -*- C++ -*-

#if !defined(eventView_H)
#define eventView_H
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/dijet.h"

namespace NtupleAna {
  const float lHC = 120;
  const float sHC = 110;
  const float rHC = lHC/sHC;
  const float dHC = 1+pow(2,rHC);

  float getDHH(float m1, float m2){
    return fabs(m1-m2*rHC)/dHC;
  }

  const float lZC = 90;
  const float sZC = lZC*sHC/lHC;

  float getXZZ(float m1, float m2){
    return pow(0.5, pow(2, (m1-lZC)/(0.1*m1) ) + pow(2, (m2-sZC)/(0.1*m2) ) );
  }

  //eventView object
  class eventView {

  public:
    dijet dijet1;
    dijet dijet2;

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
    float mZH;

    bool ZZ;

    eventView(dijet, dijet); 
    ~eventView(); 

    //void dump();
  };

}
#endif // eventView_H

