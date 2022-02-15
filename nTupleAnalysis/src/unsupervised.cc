// unsupervised_cc

#include "ZZ4b/nTupleAnalysis/interface/unsupervised.h"
// using std::cout; using std::endl;// using std::vector;
#include <vector>

std::string getROOTfileName(){
    // std::string filename = "$CMSSW_BASE/src/ZZ4b/nTupleAnalysis/data/mixedData_pull_hist_with_3b_wFvT_3bDvTMix4bDvT.root" ;
    // std::string filename = "$CMSSW_BASE/src/unsupervised/mixedData_prelim_pull_hist_corrected.root";
    // std::string filename = "$CMSSW_BASE/src/ZZ4b/nTupleAnalysis/data/mixedData_prelim_pull_hist_corrected.root";

    // std::string filename = "root://cmsxrootd.fnal.gov//store/user/smurthy/condor/ZH4b/UL/mixedData_prelim_pull_hist_corrected.root";
    std::string filename = "root://cmsxrootd.fnal.gov//store/user/smurthy/condor/ZH4b/UL/mixedData_prelim_pull_hist_no_correction.root";

    return filename;
    }

std::string getPullsFileName(){
    // std::string pullfilename = "$CMSSW_BASE/src/unsupervised/pull_cut.root" ;
    // std::string pullfilename = "$CMSSW_BASE/src/ZZ4b/nTupleAnalysis/data/pull_cut.root" ;
       std::string pullfilename = "root://cmsxrootd.fnal.gov//store/user/smurthy/condor/ZH4b/UL/pull_cut_abs.root";
    return pullfilename;}

std::vector<int> getPullPercentArr(){
    std::vector<int> pull_arr = {15,10,5,2,1};
    return pull_arr;}

// int** getM4jBinEdges(bool getPreBinned = false){
//     int preBinCount = 12;
//     int binCount = preBinCount - 2;

//     int** preM4jBinEdges = 0;
//     preM4jBinEdges = new int*[2];
//     preM4jBinEdges[0] = new int[preBinCount];
//     preM4jBinEdges[1] = new int[preBinCount];

//     int** m4jBinEdges = 0;
//     m4jBinEdges = new int*[2];
//     m4jBinEdges[0] = new int[binCount];
//     m4jBinEdges[1] = new int[binCount];
    
//     for (int lowBinEdge_ind = 0; lowBinEdge_ind < preBinCount; lowBinEdge_ind++) {
//         preM4jBinEdges[0][lowBinEdge_ind] = 250 + lowBinEdge_ind * 50;
//         preM4jBinEdges[1][lowBinEdge_ind] = 300 + lowBinEdge_ind * 50;

//         if (lowBinEdge_ind > 0 && lowBinEdge_ind < (preBinCount-1)){
//             m4jBinEdges[0][lowBinEdge_ind-1] = 250 + lowBinEdge_ind * 50;
//             m4jBinEdges[1][lowBinEdge_ind-1] = 300 + lowBinEdge_ind * 50;
//         }
//     }

//     if (getPreBinned == true){
//         return preM4jBinEdges;
//         }

//     return m4jBinEdges;
// }

// std::vector<std::vector<int>> create2DArray(){

//     std::vector<std::vector<int>> preM4jBinEdges(2, std::vector<int>(12, 0));
    
//     // for (int lowBinEdge_ind = 0; lowBinEdge_ind < preBinCount; lowBinEdge_ind++) {
//     //     preM4jBinEdges[0][lowBinEdge_ind] = lowBinEdge_ind * 50;
//     //     preM4jBinEdges[1][lowBinEdge_ind] = lowBinEdge_ind * 100;

//     // }

//     return preM4jBinEdges;
// }

std::vector<int> getM4jBinEdges(bool high = false, bool getPreBinned = false){
    int preBinCount = 12;
    int binCount = preBinCount - 2;
    
    
    if (high == true){
        if (getPreBinned == true){
            std::vector<int> m4jBinEdges(preBinCount);   
            for (int lowBinEdge_ind = 0; lowBinEdge_ind < preBinCount; lowBinEdge_ind++) {
                m4jBinEdges[lowBinEdge_ind] = 300 + lowBinEdge_ind * 50;
            }
            return m4jBinEdges;
        }

        std::vector<int> m4jBinEdges(binCount);
        for (int lowBinEdge_ind = 1; lowBinEdge_ind < preBinCount-1; lowBinEdge_ind++) {
                m4jBinEdges[lowBinEdge_ind-1] = 300 + lowBinEdge_ind * 50;
            }
        return m4jBinEdges;
    }
    
    if (getPreBinned == true){
        std::vector<int> m4jBinEdges(preBinCount);   
        for (int lowBinEdge_ind = 0; lowBinEdge_ind < preBinCount; lowBinEdge_ind++) {
            m4jBinEdges[lowBinEdge_ind] = 250 + lowBinEdge_ind * 50;
        }
    return m4jBinEdges;
    }

    std::vector<int> m4jBinEdges(binCount);
    for (int lowBinEdge_ind = 1; lowBinEdge_ind < preBinCount-1; lowBinEdge_ind++) {
        m4jBinEdges[lowBinEdge_ind-1] = 250 + lowBinEdge_ind * 50;
    }
    return m4jBinEdges;
    
}


float getLowBinIndex(float m4j){
    std::vector<int> lowBinEdge = getM4jBinEdges(false, false);
    std::vector<int> highBinEdge = getM4jBinEdges(true, false);
    float lowBinIndex = -1;
    for (int i = 0; i < static_cast<int>(lowBinEdge.size()); i++) {
        lowBinIndex = i;
        if (m4j >= lowBinEdge[i] && m4j < highBinEdge[i]){
            return lowBinIndex;
        }
    }
    return -1;
}
// testa = getM4jBinEdges(false, true);
// cout << testa.size() << "\n";
// for (int i = 0; i < testa.size(); i++){
//     cout << testa[i]<< "\n";
// }
   
