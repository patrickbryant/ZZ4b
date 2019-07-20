
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"

using namespace nTupleAnalysis;
using std::vector; using std::endl; using std::cout;


hemisphereMixTool::hemisphereMixTool(std::string name, fwlite::TFileService& fs, bool debug) {
  
  m_debug = debug;

//  dir = fs.mkdir(name);
//  unitWeight = dir.make<TH1F>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
//  unitWeight->SetCanExtend(1);
//  unitWeight->GetXaxis()->FindBin("all");
//  
//  weighted = dir.make<TH1F>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
//  weighted->SetCanExtend(1);
//  weighted->GetXaxis()->FindBin("all");
//
//  if(isMC){
//    //Make a weighted cutflow as a function of the true m4b, xaxis is m4b, yaxis is cut name. 
//    Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
//    truthM4b = dir.make<TH2F>("truthM4b", (name+"/truthM4b; Truth m_{4b} [GeV]; ;Entries").c_str(), 22, bins_m4b, 1, 1, 2);
//    truthM4b->SetCanExtend(TH1::kYaxis);
//    truthM4b->GetYaxis()->FindBin("all");
//    truthM4b->GetXaxis()->SetAlphanumeric(0);
//  }
} 

void hemisphereMixTool::addEvent(eventData* event){

  //
  //  Calculate Thrust Axis
  //
  TVector2 thrustAxis = getThrustAxis(event);

  //
  //  Make Hemispheres
  //
  // loop on jets 
  //   if projection negative one hemisphsere
  //   if projection positive other hemisphsere

  //
  // write to output tree
  //

  //
  // hemisphere wants:
  //   four vecs / original event,run / nJets / nBJets / phi_T / engineered features

  return;
}


TVector2 hemisphereMixTool::getThrustAxis(eventData* event){

  vector<TVector2> jetPts;
  for(const jetPtr& thisJet : event->selJets){
    jetPts.push_back(TVector2(thisJet->p.Px(),thisJet->p.Py()));
  }

  return calcThrust(jetPts);
}


//
// Following is From
// https://rivet.hepforge.org/code/1.8.2/a00677_source.html
//


// Do the full calculation
TVector2 hemisphereMixTool::calcThrust(const vector<TVector2>& jetPts) {

  // Make a vector of the three-momenta in the final state
  //double momentumSum(0.0);
  //for(const TVector2& p2 : jetPts) {
  //  momentumSum += p2.Mod();
  //}
  if(m_debug) cout <<  "Number of particles = " << jetPts.size() << endl;
  
  assert(jetPts.size() > 3);

  // Temporary variables for calcs
  TVector2 axis(0,0);
  double val = 0.;

  // Get thrust
  calcT(jetPts, val, axis);
  //if(m_debug) cout << "Mom sum = " << momentumSum << endl;
  //m_thrust = (val / momentumSum);
  // Make sure that thrust always points along the +ve x-axis.
  if (axis.X() < 0) axis = -1*axis;
  axis = axis.Unit();
  if(m_debug) cout << "Axis = " << axis.X() << " " << axis.Y() << endl;
  return axis;
}

// Do the general case thrust calculation
void hemisphereMixTool::calcT(const vector<TVector2>& momenta, double& t, TVector2& taxis) {
  // This function implements the iterative algorithm as described in the
  // Pythia manual. We take eight (four) different starting vectors
  // constructed from the four (three) leading particles to make sure that
  // we don't find a local maximum.
  vector<TVector2> p = momenta;
  assert(p.size() >= 3);
  unsigned int n = 4;

  vector<TVector2> tvec;
  vector<double> tval;

  for (int i = 0 ; i < 8; ++i) {
    // Create an initial vector from the leading four jets
    TVector2 foo(0,0);
    int sign = i;
    for (unsigned int k = 0 ; k < n ; ++k) {
      (sign % 2) == 1 ? foo += p[k] : foo -= p[k];
      sign /= 2;
    }
    foo=foo.Unit();

    // Iterate
    double diff=999.;
    while (diff>1e-5) {
      TVector2 foobar(0,0);
      for (unsigned int k=0 ; k<p.size() ; k++)
	(foo *p[k])>0 ? foobar+=p[k] : foobar-=p[k];
      diff=(foo-foobar.Unit()).Mod();
      foo=foobar.Unit();
    }

    // Calculate the thrust value for the vector we found
    t=0.;
    for (unsigned int k=0 ; k<p.size() ; k++)
      t+=fabs(foo*p[k]);

    // Store everything
    tval.push_back(t);
    tvec.push_back(foo);
  }

  // Pick the solution with the largest thrust
  t=0.;
  for (unsigned int i=0 ; i<tvec.size() ; i++)
    if (tval[i]>t){
      t=tval[i];
      taxis=tvec[i];
    }
}



hemisphereMixTool::~hemisphereMixTool(){} 

