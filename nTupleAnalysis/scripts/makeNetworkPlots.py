from networkPlotTools import *
import sys

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#eventClass ='SvB_ggZH2018_event'
#eventNumber='638145'#'353111'
fileName=sys.argv[1]
data = np.load(fileName)[()]
jets      = np.transpose( data['canJets']   , axes=(0,2,1) )
print(jets.shape)
otherjets = np.transpose( data['otherJets'] , axes=(0,2,1) )
# oo_weights= np.transpose( data['oo_weights'], axes=(0,3,1,2) )
# do_weights= np.transpose( data['do_weights'], axes=(0,3,1,2) )
oo_weights=               data['oo_weights']
do_weights=               data['do_weights']
dijets    = np.transpose( data['dijets']    , axes=(0,2,1) )
quadjets  = np.transpose( data['quadjets']  , axes=(0,2,1) )
q_score   =               data['q_score']
event     = np.transpose( data['event']     , axes=(0,2,1) )
c_score   =               data['c_score']


# plot=HCRPlot(jets, dijets, quadjets, q_eas, q_score, event, c_score, savefig='networkPlots/'+eventClass+eventNumber+'_network.pdf')

for e in range(jets.shape[0]):
    jv = []
    for i, jet in enumerate(jets[e,0:4]):
        jv.append( ROOT.TLorentzVector() )
        jv[i].SetPtEtaPhiM(jet[0], jet[1], jet[2], jet[3])
        jv[i].Print()

    ov = []
    for i, jet in enumerate(otherjets[e]):
        jetSel = jet[4]
        if jetSel == -1: continue
        ov.append( ROOT.TLorentzVector() )
        ov[i].SetPtEtaPhiM(jet[0], jet[1], jet[2], jet[3])
        ov[i].Print()

    print(oo_weights.shape)
    print(do_weights.shape)
    plotEvent(jv+ov, q_score[e], oo_weights=oo_weights[e], do_weights=do_weights[e], savefig='networkPlots/%s_eventDisplay_%d.pdf'%(fileName.replace('.npy',''),e))

    plotMassPlane(jv, q_score[e], savefig='networkPlots/%s_massPlane_%d.pdf'%(fileName.replace('.npy',''),e))
