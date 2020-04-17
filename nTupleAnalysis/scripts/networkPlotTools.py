import numpy as np
import matplotlib
matplotlib.use('Agg')
#%matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import ROOT

class square:
    def __init__(self, features):
        self.n    = len(features)
        self.bins = int(self.n**0.5)
        
        self.x = [i %self.bins for i in range(self.n)]
        self.y = [i//self.bins for i in range(self.n)]
        self.weights = features  

class HCRPlot:
    def __init__(self, jets, dijets, quadjets, q_eas, q_score, event, output, savefig='test.pdf'):
        self.savefig = savefig
        self.fig = plt.figure(figsize=[9, 4])
        self.cmap = 'BuPu'
    
        gs = self.fig.add_gridspec(nrows=6, ncols=12)

        jetIndex = [1,2,3,4,
                    1,3,2,4,
                    1,4,2,3]
        self.axjs = []
        for j in range(12):
            self.axjs.append(self.fig.add_subplot(gs[0,j]))
            self.axjs[j].axis('off')
            self.axjs[j].set_aspect(1)
            self.axjs[j].set_title(jetIndex[j])
            
        plt.text(-0.5, 0.5, 'jets', va='center', ha='right', transform=self.axjs[0].transAxes)

        self.axds = []
        for d in range(6):
            self.axds.append(self.fig.add_subplot(gs[1, 2*d:2*(d+1)]))
            self.axds[d].axis('off')
            self.axds[d].set_aspect(1)

        plt.text(-0.5, 0.5, 'dijets', va='center', ha='right', transform=self.axds[0].transAxes)
        #plt.text(0, 1-2/6, 'dijets', va='center', ha='right', transform=self.fig.transFigure)

        self.axqs = []
        for q in range(3):
            self.axqs.append(self.fig.add_subplot(gs[2, 4*q:4*(q+1)]))
            self.axqs[q].axis('off')
            self.axqs[q].set_aspect(1)

        plt.text(-0.5, 0.5, 'quadjets', va='center', ha='right', transform=self.axqs[0].transAxes)
        #plt.text(0, 1-3/6, 'quadjets', va='center', ha='right', transform=self.fig.transFigure)
            
        self.axqeas = []
        for q in range(3):
            self.axqeas.append(self.fig.add_subplot(gs[3, 4*q:4*(q+1)]))         
            self.axqeas[q].set_aspect(1)
            self.axqeas[q].spines['top'].set_color(None)
            self.axqeas[q].spines['left'].set_color(None)
            self.axqeas[q].spines['bottom'].set_color(None)
            self.axqeas[q].spines['right'].set_color(None)
            self.axqeas[q].tick_params(bottom=False,left=False, labelbottom=False, labelleft=False)
            self.axqeas[q].patch.set_edgecolor((0,0,0,q_score[q]))  
            self.axqeas[q].patch.set_linewidth(2)

        plt.text(-0.5, 0.5, 'ancilllary\n features added', va='center', ha='right', transform=self.axqeas[0].transAxes)
        #plt.text(0, 1-4/6, 'ancillary features added', va='center', ha='right', transform=self.fig.transFigure)
            
        self.axe = self.fig.add_subplot(gs[4,0:12])
        self.axe.axis('off')
        self.axe.set_aspect(1)

        plt.text(-0.5, 0.5, 'event', va='center', ha='right', transform=self.axe.transAxes)

        self.axo = self.fig.add_subplot(gs[5,5:7])
        self.axo.axis('off')
        self.axo.set_aspect(1)

        plt.text(0.5, -0.5, 'P(ZZ, ZH, multijet, ttbar)', va='center', ha='center', transform=self.axo.transAxes)
        #plt.text(0, 1-5/6, 'event', va='center', ha='right', transform=self.fig.transFigure)
        
        self.jets = []
        for j, jet in enumerate(jets):
            self.jets.append(square(jet))
            hist, xBins, yBins, mesh = self.axjs[j].hist2d(self.jets[j].x,
                                                           self.jets[j].y, 
                                                           weights=self.jets[j].weights,
                                                           bins=self.jets[j].bins,
                                                           cmap=self.cmap,
                                                           cmin=jets.min(), cmax=jets.max(),
                                                          )
        self.dijets = []
        for d, dijet in enumerate(dijets):
            self.dijets.append(square(dijet))
            hist, xBins, yBins, mesh = self.axds[d].hist2d(self.dijets[d].x, 
                                                           self.dijets[d].y, 
                                                           weights=self.dijets[d].weights, 
                                                           bins=self.dijets[d].bins,
                                                           cmap=self.cmap,
                                                           cmin=dijets.min(), cmax=dijets.max(),
                                                          )
        self.quadjets = []
        for q, quadjet in enumerate(quadjets):
            self.quadjets.append(square(quadjet))
            hist, xBins, yBins, mesh = self.axqs[q].hist2d(self.quadjets[q].x, 
                                                           self.quadjets[q].y, 
                                                           weights=self.quadjets[q].weights,
                                                           bins=self.quadjets[q].bins,
                                                           cmap=self.cmap,
                                                           cmin=quadjets.min(), cmax=quadjets.max(),
                                                          )
        self.q_eas = []
        for q, q_ea in enumerate(q_eas):
            self.q_eas.append(square(q_ea))
            hist, xBins, yBins, mesh = self.axqeas[q].hist2d(self.q_eas[q].x, 
                                                             self.q_eas[q].y, 
                                                             weights=self.q_eas[q].weights,
                                                             bins=self.q_eas[q].bins,
                                                             cmap=self.cmap,
                                                             cmin=q_eas.min(), cmax=q_eas.max(),
                                                             )   
        self.event = square(event)
        hist, xBins, yBins, mesh = self.axe.hist2d(self.event.x,
                                                   self.event.y,
                                                   weights=self.event.weights,
                                                   bins=self.event.bins,
                                                   cmap=self.cmap,
                                                  )
        outx = [i for i in range(len(output))]
        outy = [0 for i in range(len(output))]
        hist, xBins, yBins, mesh = self.axo.hist2d(outx,
                                                   outy,
                                                   weights=output,
                                                   bins=[len(output),1],
                                                   cmap=self.cmap,
                                                  )
        #plt.colorbar(mesh, self.axo)
        print(savefig)
        self.fig.savefig(self.savefig)
        
            

def plotEvent(TLorentzVectors, q_score=None, savefig='test.pdf'):
    fig=plt.figure()
    
    pts  = np.array([v.Pt()        for v in TLorentzVectors])
    etas = np.array([v.Eta()       for v in TLorentzVectors])
    phis = np.array([v.Phi()/np.pi for v in TLorentzVectors])
    ms   = np.array([v.M()         for v in TLorentzVectors])
    
    plt.scatter(etas, phis, marker='o', s=5*ms, c=pts, lw=1, zorder=10, cmap='BuPu', edgecolors='black')
    cbar = plt.colorbar(label='$p_{T}$ [GeV]', 
                        #use_gridspec=False, location="top"
                       )
    
    dijet_mass_scale = 50
    if q_score is not None:
        quadjets = [[(0,1),(2,3)],
                    [(0,2),(1,3)],
                    [(0,3),(1,2)]]
        for q, quadjet in enumerate(quadjets):
            for dijet in quadjet:
                dijet_mass = (TLorentzVectors[dijet[0]] + TLorentzVectors[dijet[1]]).M()
                plt.plot(etas[dijet,], phis[dijet,], '-', 
                         lw=dijet_mass/dijet_mass_scale,
                         color='black', alpha=q_score[q])
    
    lt = plt.scatter([],[], s=0,    lw=0, edgecolors='none',  facecolors='none')
    l1 = plt.scatter([],[], s=5*5,  lw=1, edgecolors='black', facecolors='none')
    l2 = plt.scatter([],[], s=5*10, lw=1, edgecolors='black', facecolors='none')
    l3 = plt.scatter([],[], s=5*15, lw=1, edgecolors='black', facecolors='none')
    l4 = plt.scatter([],[], s=5*20, lw=1, edgecolors='black', facecolors='none')
    
    dt, = plt.plot([],[], '-', lw=0,                    color='none')
    d1, = plt.plot([],[], '-', lw= 50/dijet_mass_scale, color='black')
    d2, = plt.plot([],[], '-', lw=100/dijet_mass_scale, color='black')
    d3, = plt.plot([],[], '-', lw=150/dijet_mass_scale, color='black')
    d4, = plt.plot([],[], '-', lw=200/dijet_mass_scale, color='black')

    #labels = [  "jet mass [GeV]",  "5",  "10",  "15",  "20",
    #          "dijet mass [GeV]", "50", "100", "150", "200"]
    handles = [l1, d1,
               l2, d2,
               l3, d3,
               l4, d4,
               lt, dt]
    labels = [ "5",  "50",
              "10", "100",
              "15", "150",
              "20", "200",
              "jet mass [GeV]", "dijet mass [GeV]",
             ]

    leg = plt.legend(handles, labels, 
                     ncol=5, 
                     frameon=False, #fancybox=False, edgecolor='black',
                     #markerfirst=False,
                     bbox_to_anchor=(0, 1), loc='lower left',
                     handlelength=0.8, #handletextpad=1, #borderpad = 0.5,
                     #title='mass [GeV]', 
                     columnspacing=1.8,
                     scatterpoints = 1, scatteryoffsets=[0.5])
    
    # get the width of your widest label, since every label will need 
    # to shift by this amount after we align to the right
    renderer = fig.canvas.get_renderer()
    #shift = max([t.get_window_extent(renderer).width for t in leg.get_texts()])
    shift=17
    for t in leg.get_texts():
        t.set_ha('right') # ha is alias for horizontalalignment
        t.set_position((shift,0))

    leg.get_texts()[-2].set_position((55,0))
    leg.get_texts()[-1].set_position((55,0))
                
    # plot settings
    plt.xlim(-2.5, 2.5); plt.ylim(-1, 1)
    plt.xlabel('$\eta$'); plt.ylabel('$\phi$ [$\pi$]')
    plt.xticks(np.linspace(-2.5, 2.5, 5)); plt.yticks(np.linspace(-1, 1, 5))


    print(savefig)
    fig.savefig(savefig)



def plotMassPlane(TLorentzVectors, q_score, savefig='test.pdf'):
    fig, ax = plt.subplots(figsize=[5,4])
    ax.set_aspect(1)
    
    quadjets = [[(0,1),(2,3)],
                [(0,2),(1,3)],
                [(0,3),(1,2)]]
    m1s = []
    m2s = []
    for q, quadjet in enumerate(quadjets):
        dijet1 = quadjet[0]
        dijet2 = quadjet[1]
        dijet1 = TLorentzVectors[dijet1[0]] + TLorentzVectors[dijet1[1]]
        dijet2 = TLorentzVectors[dijet2[0]] + TLorentzVectors[dijet2[1]]
        if dijet1.M() > dijet2.M():
            m1s.append(dijet1.M())
            m2s.append(dijet2.M())
        else:
            m1s.append(dijet2.M())
            m2s.append(dijet1.M())

   
    ms = plt.scatter(m1s, m2s, marker='o', s=25, c=q_score, lw=1, zorder=10, cmap='Greys', vmin=0, vmax=1, edgecolors='black')
    cbar = plt.colorbar(label='quadjet score',
                        #use_gridspec=False, location="top"
                       ) 

    #diagonal line x=y
    plt.plot([0,500],[0,500],'--',color='grey', lw=0.5)

    lt = plt.scatter([],[], s=0,    lw=0, edgecolors='none',  facecolors='none')
    zz = plt.scatter([ 91.0], [ 87.2], marker='x', s=10, c='g', lw=1, label='ZZ')
    zh = plt.scatter([123.0], [ 92.0], marker='x', s=10, c='r', lw=1, label='ZH')
    hh = plt.scatter([120.0], [115.0], marker='x', s=10, c='b', lw=1, label='HH')

    handles = [lt, zz, zh, hh]
    labels  = ['Signal Region Center:', 'ZZ', 'ZH', 'HH']
    
    leg = plt.legend(handles, labels, 
                     ncol=4, 
                     frameon=False, #fancybox=False, edgecolor='black',
                     #markerfirst=False,
                     bbox_to_anchor=(1, 1), loc='lower right',
                     handlelength=0, #handletextpad=1, #borderpad = 0.5,
                     borderpad=0,
                     #title='Signal Region Centers', 
                     columnspacing=1.2,
                     scatterpoints = 1, scatteryoffsets=[0.5])
                
    # plot settings
    plt.xlim(0, 500); plt.ylim(0, 500)
    plt.xlabel('leading mass dijet mass [GeV]'); plt.ylabel('subleading mass dijet mass [GeV]')
    plt.xticks(np.linspace(0, 500, 6)); plt.yticks(np.linspace(0, 500, 6))
    
    axins = zoomed_inset_axes(ax,3, loc='upper left') # zoom = 6
    
    #now have to plot everything again becaue matplotlib is horribly structured
    plt.scatter(m1s, m2s, marker='o', s=25, c=q_score, lw=1, zorder=10, cmap='Greys', vmin=0, vmax=1, edgecolors='black')
    plt.plot([0,500],[0,500],'--',color='grey', lw=0.5)
    zz = plt.scatter([ 91.0], [ 87.2], marker='x', s=10, c='g', lw=1, label='ZZ')
    zh = plt.scatter([123.0], [ 92.0], marker='x', s=10, c='r', lw=1, label='ZH')
    hh = plt.scatter([120.0], [115.0], marker='x', s=10, c='b', lw=1, label='HH')
    
    # sub region of the original image
    x1, y1, x2, y2 = 67+5, 67-5, 145+5, 145-5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(ticks=[90,110,130], visible=True)
    plt.yticks(ticks=[90,110,130], visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.5)
    
    plt.draw()

    print(savefig)
    fig.savefig(savefig)
