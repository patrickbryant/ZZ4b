// -*- C++ -*-OB

#if !defined(eventView_H)
#define eventView_H
#include <iostream>
#include <TLorentzVector.h>
#include "nTupleAnalysis/baseClasses/interface/dijet.h"

namespace nTupleAnalysis {
  //eventView object
  class eventView {

    //DiJet Mass Plane Region Definitions
    const float leadStBias = 1.02; //leading st dijet mass peak is shifted up by a few percent
    const float sublStBias = 0.98; //sub-leading st dijet mass peak is shifted down by a few percent
    const float mZ =  91.0;
    const float mH = 125.0;
    const float leadH = 127.50; // mH * leadStBias
    const float sublH = 122.50; // mH * sublStBias
    const float leadZ =  92.82; // mZ * leadStBias
    const float sublZ =  89.18; // mZ * sublStBias

    const float xMaxZZSR =  2.60;
    const float rMaxZZCR = 28.00;
    const float    sZZCR =  1.01;
    const float rMaxZZSB = 40.00;
    const float    sZZSB =  1.02;

    const float xMaxZHSR =  1.90;
    const float rMaxZHCR = 30.00;
    const float    sZHCR =  1.04;
    const float rMaxZHSB = 45.00;
    const float    sZHSB =  1.06;

    const float xMaxHHSR =  1.90;
    const float rMaxHHCR = 30.00;
    const float    sHHCR =  1.04;
    const float rMaxHHSB = 45.00;
    const float    sHHSB =  1.06;

    const float slopeDBB = leadStBias/sublStBias;
    const float denomDBB = sqrt(1+pow(slopeDBB, 2));

    float getXZZ(float m1, float m2){
      float sigmaLead = (m1-leadZ)/(0.1*m1);
      float sigmaSubl = (m2-sublZ)/(0.1*m2);
      float xZZ2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xZZ2);
    }

    float leadZH; float sublZH;
    float getXZH(float m1, float m2){//need to consider case where m1 is H and m2 is Z but also m1 is Z and m2 is H
      //case where m1 is H and m2 is Z
      float sigmaLeadNormal   = (m1-leadH)/(0.1*m1);
      float sigmaSublNormal   = (m2-sublZ)/(0.1*m2);
      //case where m1 is Z and m2 is H
      float sigmaLeadInverted = (m1-leadZ)/(0.1*m1);
      float sigmaSublInverted = (m2-sublH)/(0.1*m2);

      float xZH2Normal   = pow(sigmaLeadNormal,   2) + pow(sigmaSublNormal,   2);
      float xZH2Inverted = pow(sigmaLeadInverted, 2) + pow(sigmaSublInverted, 2);

      if(xZH2Normal > xZH2Inverted){ //Inverted mass order is better match
	leadZH = leadZ;
	sublZH = sublH;
	return sqrt(xZH2Inverted);
      }else{ //Normal mass order is better match
	leadZH = leadH;
	sublZH = sublZ;
	return sqrt(xZH2Normal);
      }
    }

    float getXHH(float m1, float m2){
      float sigmaLead = (m1-leadH)/(0.1*m1);
      float sigmaSubl = (m2-sublH)/(0.1*m2);
      float xHH2 = pow(sigmaLead, 2) + pow(sigmaSubl, 2);
      return sqrt(xHH2);
    }

    float getDBB(float m1, float m2){
      //float slopeDZH = leadZH/sublZH;
      //float denomDZH = sqrt(1+pow(slopeDZH, 2));
      //float DZHorHZ = fabs(m1-m2*slopeDZH)/denomDZH;
      //float DZZorHH = fabs(m1-m2*slopeDBB)/denomDBB;
      //return std::min(DZHorHZ, DZZorHH);
      return fabs(m1-m2*slopeDBB)/denomDBB;
    }


  public:

    std::shared_ptr<nTupleAnalysis::dijet> lead;
    std::shared_ptr<nTupleAnalysis::dijet> subl;

    std::shared_ptr<nTupleAnalysis::dijet> leadSt;
    std::shared_ptr<nTupleAnalysis::dijet> sublSt;

    std::shared_ptr<nTupleAnalysis::dijet> leadM;
    std::shared_ptr<nTupleAnalysis::dijet> sublM;

    std::shared_ptr<nTupleAnalysis::dijet> close;
    std::shared_ptr<nTupleAnalysis::dijet> other;

    bool truthMatch = false;

    TLorentzVector p;
    float pt;
    float eta;
    float phi;
    float m;
    float e;

    float m4j;

    float dBB;

    float mZZ;
    float mZH;
    float mHH;

    float xZZ;
    float xZH;
    float xHH;
    bool ZZSR;
    bool ZHSR;
    bool HHSR;
    bool SR;

    float rZZCR;
    float rZHCR;
    float rHHCR;
    bool ZZCR;
    bool ZHCR;
    bool HHCR;
    bool CR;

    float rZZSB;
    float rZHSB;
    float rHHSB;
    bool ZZSB;
    bool ZHSB;
    bool HHSB;
    bool SB;

    //m4j dependent view requirements (MDRs)
    bool passLeadStMDR;
    bool passSublStMDR;
    bool passMDRs;

    //m4j dependent cuts (MDCs)
    //bool passLeadMDC;
    //bool passSublMDC;
    //bool passMDCs;

    float dEtaBB;
    float dRBB;
    //bool passDEtaBB;

    float FvT_q_score;
    float SvB_q_score;
    float SvB_MA_q_score;

    // Need for unsuperised analysis
    float SRvsSB_pull  = -99;
    float SRvsSB_ratio = -99;

    eventView(std::shared_ptr<nTupleAnalysis::dijet>&, std::shared_ptr<nTupleAnalysis::dijet>&, float FvT_q_score_ = -99, float SvB_q_score_ = -99, float SvB_MA_q_score_ = -99); 
    ~eventView(); 

    //void dump();
  };

}
#endif // eventView_H

