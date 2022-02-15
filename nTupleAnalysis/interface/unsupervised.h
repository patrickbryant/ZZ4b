// unsupervised_h
// Things to put in this file:
// Location of saved preliminary file
// Save the pull value array as a ROOT histogram and save it here
// Write the functions that call for loops for binning and percentage values here in this file
// For reference on how to make pull cut histograms, look in unsupervised/make_cut_hists.py 
// Remake and save in make_clean_pull_hist.cpp (Delete this file?)


#if !defined(unsupervised_H)
#define unsupervised_H
 #include <vector>
 #include <string>
 #include <TFile.h>
 #include <TH2F.h>

    std::string getROOTfileName();

    // float getPullVal(int percentInd, int m4jBinInd);
    std::string getPullsFileName();
    std::vector<int> getPullPercentArr();
    int** getM4jBinEdges(bool getPreBinned );
    std::vector<int> getM4jBinEdges(bool high , bool getPreBinned );
    float getLowBinIndex(float m4j);


#endif // unsupervised_H
