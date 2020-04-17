from networkPlotTools import *

eventClass ='SvB_ggZH2018_event'
eventNumber='638145'#'353111'
fileName=eventClass+eventNumber+'.npy'
data = np.load(fileName)[()]
jets      = np.transpose( data['canJets'] )
otherjets = np.transpose( data['otherJets'] )
dijets    = np.transpose( data['dijets'] )
quadjets  = np.transpose( data['quadjets_sym'] )
q_eas     = np.transpose( data['quadjets_sym_eventAncillary'] )
q_score   =               data['q_score'][0]
print(q_score)
event     = np.transpose( data['event'] )[0]
output    = np.transpose( data['classProb'] )

plot=HCRPlot(jets, dijets, quadjets, q_eas, q_score, event, output, savefig='networkPlots/'+eventClass+eventNumber+'_network.pdf')


ptScale,  ptMean  = 43.64330444, 115.14173714
etaScale, etaMean = 2.4, 0
phiScale, phiMean = np.pi, 0
mScale,   mMean   = 6.15257094, 16.08879767

jv = []
for i, jet in enumerate(jets[0:4]):
    jv.append( ROOT.TLorentzVector() )
    jv[i].SetPtEtaPhiM(jet[0]*ptScale + ptMean, jet[1]*etaScale, jet[2]*phiScale, jet[3]*mScale + mMean)
    
ov = []
for i, jet in enumerate(otherjets):
    pt = jet[0]*ptScale + ptMean
    if pt < 1e-3: continue
    ov.append( ROOT.TLorentzVector() )
    ov[i].SetPtEtaPhiM(jet[0]*ptScale + ptMean, jet[1]*etaScale, jet[2]*phiScale, jet[3]*mScale + mMean)


plotEvent(jv, q_score, savefig='networkPlots/'+eventClass+eventNumber+'_eventDisplay.pdf')

plotMassPlane(jv, q_score, savefig='networkPlots/'+eventClass+eventNumber+'_massPlane.pdf')
