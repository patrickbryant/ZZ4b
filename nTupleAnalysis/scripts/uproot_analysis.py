# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh
# source /cvmfs/sft.cern.ch/lcg/nightlies/dev4/Wed/coffea/0.7.13/x86_64-centos7-gcc10-opt/coffea-env.sh 
import pickle, os, time
from dataclasses import dataclass
import awkward as ak
import numpy as np
import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
NanoAODSchema.warn_missing_crossrefs = False
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
from coffea import hist, processor
#from coffea.btag_tools import BTagScaleFactor
import correctionlib
#from coffea.lookup_tools.json_converters import convert_correctionlib_file
#from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from MultiClassifierSchema import MultiClassifierSchema
from functools import partial

@dataclass
class variable:
    def __init__(self, name, bins):
        self.name = name
        self.bins = bins

class analysis(processor.ProcessorABC):
    def __init__(self):
        print('Initialize Analysis Processor')
        self.cuts = ['passPreSel']
        self.tags = ['fourTag']
        self.regions = ['SR']
        self.variables = [variable('SvB_ps_hh', hist.Bin('x', 'SvB Regressed P(Signal) | P(HH) is largest', 20, 0, 1)),
                          variable('SvB_ps_zh', hist.Bin('x', 'SvB Regressed P(Signal) | P(ZH) is largest', 20, 0, 1)),
                          variable('SvB_ps_zz', hist.Bin('x', 'SvB Regressed P(Signal) | P(ZZ) is largest', 20, 0, 1)),
        ]

        self._accumulator = processor.dict_accumulator({'cutflow': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, float)),
                                                        'nEvent' : processor.defaultdict_accumulator(int),
                                                        'hists'  : processor.dict_accumulator()})

        self.nHists = 0

        for cut in self.cuts:
            print(f'Making Hists for {cut}')
            self._accumulator['hists'][cut] = processor.dict_accumulator()
            for tag in self.tags:
                print(f'    {tag}')
                self._accumulator['hists'][cut][tag] = processor.dict_accumulator()
                for region in self.regions:
                    print(f'        {region}')
                    self._accumulator['hists'][cut][tag][region] = processor.dict_accumulator()
                    for var in self.variables:
                        self._accumulator['hists'][cut][tag][region][var.name] = hist.Hist('Events',
                                                                                           hist.Cat('trigWeight', 'Trigger Weight'),
                                                                                           var.bins)
                    self.nHists += len(self._accumulator['hists'][cut][tag][region])
        print(f'{self.nHists} total histograms')
        
    @property
    def accumulator(self):
        return self._accumulator

    def cutflow(self, output, event, cut, allTag=False):
        if allTag:
            w = event.weight
            sumw = np.sum(w)
            sumw_3, sumw_4 = sumw, sumw
        else:
            e3, e4 = event[event.threeTag], event[event.fourTag]
            sumw_3 = np.sum(e3.weight)
            sumw_4 = np.sum(e4.weight)

        output['cutflow']['threeTag'][cut] += sumw_3
        output['cutflow'][ 'fourTag'][cut] += sumw_4

        if event.metadata.get('isMC', False):
            output['cutflow']['threeTag'][cut+'_HLT_Bool'] += np.sum(e3.weight*e3.passHLT)
            output['cutflow'][ 'fourTag'][cut+'_HLT_Bool'] += np.sum(e4.weight*e4.passHLT)

            output['cutflow']['threeTag'][cut+'_HLT_MC'  ] += np.sum(e3.weight*e3.trigWeight.MC)
            output['cutflow'][ 'fourTag'][cut+'_HLT_MC'  ] += np.sum(e4.weight*e4.trigWeight.MC)

            output['cutflow']['threeTag'][cut+'_HLT_Data'] += np.sum(e3.weight*e3.trigWeight.Data)
            output['cutflow'][ 'fourTag'][cut+'_HLT_Data'] += np.sum(e4.weight*e4.trigWeight.Data)
            

    def process(self, event):
        tstart = time.time()
        output = self.accumulator.identity()

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        year    = event.metadata['year']
        isMC    = event.metadata.get('isMC',  False)
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        btagSF  = event.metadata.get('btagSF', None)
        nEvent = len(event)
        output['nEvent'][dataset] += nEvent

        if isMC:
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            if btagSF:
                #btagSF = BTagScaleFactor(btagSF, 3, methods='iterativefit') # 0: loose, 1: medium, 2: tight, 3: shape
                btagSF = correctionlib.CorrectionSet.from_file(btagSF)['deepJet_shape']
                #btagSF = convert_correctionlib_file(btagSF)[('deepJet_shape', 'correctionlib_wrapper')][0]
                print(btagSF)


        print(fname)
        print(f'Process {nEvent} Events ({estart} to {estop})')

        path = fname.replace(fname.split('/')[-1],'')
        SvB    = NanoEventsFactory.from_root(f'{path}SvB.root',    entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB
        SvB_MA = NanoEventsFactory.from_root(f'{path}SvB_MA.root', entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB_MA
        event['SvB']    = SvB
        event['SvB_MA'] = SvB_MA

        if isMC:
            output['cutflow']['threeTag']['all'] = lumi * xs * kFactor    
            output['cutflow'][ 'fourTag']['all'] = lumi * xs * kFactor    
        else:
            self.cutflow(output, event, 'all', allTag = True)


        # Get trigger decisions 
        if year == 2016:
            event['passHLT'] = event.HLT.QuadJet45_TripleBTagCSV_p087 | event.HLT.DoubleJet90_Double30_TripleBTagCSV_p087 | event.HLT.DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6
        if year == 2017:
            event['passHLT'] = event.HLT.PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0 | event.HLT.DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33
        if year == 2018:
            event['passHLT'] = event.HLT.DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 | event.HLT.PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5

        if not isMC: # for data, apply trigger cut first thing, for MC, keep all events and apply trigger in cutflow and for plotting
            event = event[event.passHLT]


        # Preselection
        event['Jet', 'selected'] = (event.Jet.pt>=40) & (np.abs(event.Jet.eta)<=2.4) & ~((event.Jet.puId<0b110)&(event.Jet.pt<50))
        event['nJet_selected']   = ak.num(event.Jet[event.Jet.selected])

        event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagDeepFlavB>=0.6)
        event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagDeepFlavB>=0.3)
        event['nJet_tagged']         = ak.num(event.Jet[event.Jet.tagged])
        event['nJet_tagged_loose']   = ak.num(event.Jet[event.Jet.tagged_loose])


        # check that coffea analysis agrees with c++
        fourTag  = (event.nJet_tagged       >= 4)
        threeTag = (event.nJet_tagged_loose == 3) & (event.nJet_selected >= 4)
        event['issue'] = (threeTag!=event.threeTag)|(fourTag!=event.fourTag)
        if ak.any(event.issue):
            print('WARNING: selected jets or fourtag calc not equal to picoAOD values')

        event[ 'fourTag'] =  fourTag
        event['threeTag'] = threeTag

        # keep only three or four tag event
        event = event[event.threeTag | event.fourTag]

        if isMC:
            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)

        self.cutflow(output, event, 'passPreSel')

        # Build and select boson candidate jets with bRegCorr applied
        event['selJet'] = event.Jet[event.Jet.selected]
        canJet = event.selJet * event.selJet.bRegCorr
        canJet['btagDeepFlavB'] = event.selJet.btagDeepFlavB
        canJet['hadronFlavour'] = event.selJet.hadronFlavour


        # sort by btag score
        canJet = canJet[ak.argsort(canJet.btagDeepFlavB, axis=1, ascending=False)]
        # take top four
        canJet = canJet[:,0:4]
        # return to pt sorting 
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        event['canJet'] = canJet
        event['v4j'] = canJet.sum(axis=1)


        # Build dijets, indexed by dijet[event,pairing,0/1]
        pairing = [([0,2],[0,1],[0,1]),
                    ([1,3],[2,3],[3,2])]
        dijet       = canJet[:,pairing[0]]     +   canJet[:,pairing[1]]
        dijet['st'] = canJet[:,pairing[0]].pt  +   canJet[:,pairing[1]].pt
        dijet['dr'] = canJet[:,pairing[0]].delta_r(canJet[:,pairing[1]])
        dijet['lead'] = canJet[:,pairing[0]]
        dijet['subl'] = canJet[:,pairing[1]]
        # Sort dijets within views to be lead st, subl st
        dijet = dijet[ak.argsort(dijet.st, axis=2, ascending=False)]
        # Now indexed by dijet[event,pairing,lead/subl st]


        # Compute dijetMass cut with independent min/max for lead/subl 
        minDijetMass = np.array([[[ 52, 50]]])
        maxDijetMass = np.array([[[180,173]]])
        dijet['passDijetMass'] = (minDijetMass < dijet.mass) & (dijet.mass < maxDijetMass)


        # Compute MDRs
        min_m4j_scale = np.array([[ 360, 235]])
        min_dr_offset = np.array([[-0.5, 0.0]])
        max_m4j_scale = np.array([[ 650, 650]])
        max_dr_offset = np.array([[ 0.5, 0.7]])
        max_dr        = np.array([[ 1.5, 1.5]])
        m4j = np.repeat(np.reshape(np.array(event.v4j.mass), (-1,1,1)), 2, axis=2)
        dijet['passMDR'] = (min_m4j_scale/m4j + min_dr_offset < dijet.dr) & (dijet.dr < np.maximum(max_m4j_scale/m4j + max_dr_offset, max_dr))


        # Compute consistency of dijet masses with boson masses
        mZ =  91.0
        mH = 125.0
        st_bias = np.array([[[1.02, 0.98]]])
        cZ = mZ * st_bias
        cH = mH * st_bias

        dijet['xZ'] = (dijet.mass - cZ)/(0.1*dijet.mass)
        dijet['xH'] = (dijet.mass - cH)/(0.1*dijet.mass)


        # Build quadjets
        np.random.seed(0)
        quadjet = ak.zip({'lead': dijet[:,:,0],
                          'subl': dijet[:,:,1],
                          'passDijetMass': ak.all(dijet.passDijetMass, axis=2),
                          'random': np.random.uniform(low=0.1, high=0.9, size=(dijet.__len__(), 3))
                      }, with_name='quadjet')
        quadjet['SvB_q_score'] = np.concatenate((np.reshape(np.array(event.SvB.q_1234), (-1,1)), 
                                                 np.reshape(np.array(event.SvB.q_1324), (-1,1)),
                                                 np.reshape(np.array(event.SvB.q_1423), (-1,1))), axis=1)
        quadjet['SvB_MA_q_score'] = np.concatenate((np.reshape(np.array(event.SvB_MA.q_1234), (-1,1)), 
                                                    np.reshape(np.array(event.SvB_MA.q_1324), (-1,1)),
                                                    np.reshape(np.array(event.SvB_MA.q_1423), (-1,1))), axis=1)

        # Compute Signal Regions
        quadjet['xZZ'] = np.sqrt(quadjet.lead.xZ**2 + quadjet.subl.xZ**2)
        quadjet['xHH'] = np.sqrt(quadjet.lead.xH**2 + quadjet.subl.xH**2)
        quadjet['xZH'] = np.sqrt(np.minimum(quadjet.lead.xH**2 + quadjet.subl.xZ**2, 
                                            quadjet.lead.xZ**2 + quadjet.subl.xH**2))
        max_xZZ = 2.6
        max_xZH = 1.9
        max_xHH = 1.9
        quadjet['ZZSR'] = quadjet.xZZ < max_xZZ
        quadjet['ZHSR'] = quadjet.xZH < max_xZH
        quadjet['HHSR'] = quadjet.xHH < max_xHH
        quadjet['SR'] = quadjet.ZZSR | quadjet.ZHSR | quadjet.HHSR
        quadjet['SB'] = quadjet.passDijetMass & ~quadjet.SR


        # pick quadjet at random giving preference to ones which passDijetMass and MDRs
        quadjet['rank'] = 10*quadjet.passDijetMass + quadjet.lead.passMDR + quadjet.subl.passMDR + quadjet.random
        quadjet['selected'] = quadjet.rank == np.max(quadjet.rank, axis=1)

        event[  'dijet'] =   dijet
        event['quadjet'] = quadjet
        event['quadjet_selected'] = quadjet[quadjet.selected][:,0]

        event.issue = event.passDijetMass != event.quadjet_selected.passDijetMass
        if ak.any(event.issue):
            print('WARNING: passDijetMass calc not equal to picoAOD value')


        if btagSF is not None:
            # btagSF evaluation causes a significant performance hit (50k/s to 7k/s)
            # SF = btagSF.eval('central', 
            #                  event.canJet.hadronFlavour, abs(event.canJet.eta), event.canJet.pt, event.canJet.btagDeepFlavB, 
            #                  ignore_missing=True)
            cj, nj = ak.flatten(event.canJet), ak.num(event.canJet)
            SF = btagSF.evaluate('central', np.array(cj.hadronFlavour), np.array(abs(cj.eta)), np.array(cj.pt), np.array(cj.btagDeepFlavB))
            SF = ak.unflatten(SF, nj)
            

            event['btagSF_central'] = np.prod(SF, axis=1)
            event.weight = event.weight * event.btagSF_central


        self.cutflow(output, event[event.quadjet_selected.passDijetMass], 'passDijetMass')
        self.cutflow(output, event[event.quadjet_selected.SR], 'SR')

        event['SvB', 'passMinPs'] = (event.SvB.pzz>0.01) | (event.SvB.pzh>0.01) | (event.SvB.phh>0.01) 
        event['SvB', 'zz'] = (event.SvB.pzz >  event.SvB.pzh) & (event.SvB.pzz >  event.SvB.phh)
        event['SvB', 'zh'] = (event.SvB.pzh >  event.SvB.pzz) & (event.SvB.pzh >  event.SvB.phh)
        event['SvB', 'hh'] = (event.SvB.phh >= event.SvB.pzz) & (event.SvB.phh >= event.SvB.pzh)

        mask = event.fourTag
        mask = mask & event.quadjet_selected.SR
        mask = mask & event.SvB.passMinPs
        mask = mask & event.SvB.hh
        hist_event = event[mask]

        weight_no_trig = hist_event.genWeight * (lumi * xs * kFactor / genEventSumw)
        output['hists']['passPreSel']['fourTag']['SR']['SvB_ps_hh'].fill(trigWeight='Boolean', x=hist_event.SvB.ps, weight=weight_no_trig * hist_event.passHLT)
        output['hists']['passPreSel']['fourTag']['SR']['SvB_ps_hh'].fill(trigWeight='MC',      x=hist_event.SvB.ps, weight=weight_no_trig * hist_event.trigWeight.MC)
        output['hists']['passPreSel']['fourTag']['SR']['SvB_ps_hh'].fill(trigWeight='Data',    x=hist_event.SvB.ps, weight=weight_no_trig * hist_event.trigWeight.Data)

        elapsed = time.time() - tstart
        print(f'{nEvent/elapsed:,.0f} events/s')
        return output

    def postprocess(self, accumulator):
        return accumulator



if __name__ == '__main__':
    eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor'
    nfs_base = '/uscms/home/bryantp/nobackup/ZZ4b'
    #nfs_base = os.path.expanduser(nfs_base)
    eos = False

    year = 2018
    dataset = f'ZZ4b{year}'
    input_path = f'{eos_base if eos else nfs_base}/{dataset}'
    output_path = f'{nfs_base}/{dataset}'

    # metadata = {'isMC'  : True,
    #             'xs'    : 0.03105*0.5824**2,
    #             'lumi'  : 36.3e3,
    #             'year'  : 2016,
    #             'btagSF': 'nTupleAnalysis/baseClasses/data/BTagSF2016/DeepJet_2016LegacySF_V1.csv.gz',
    # }
    # fileset = {'HH4b2016': {'files': [f'{input_path}/picoAOD.root'],
    #                         'metadata': metadata},
    # }
    metadata = {'isMC'  : True,
                'xs'    : 15.5*0.1512*0.1512,
                'lumi'  : 59.8e3,
                'year'  : year,
                'btagSF': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz',
    }
    fileset = {dataset: {'files': [f'{input_path}/picoAOD.root'],
                         'metadata': metadata},
    }

    tstart = time.time()
    output = processor.run_uproot_job(
        fileset,
        treename='Events',
        processor_instance=analysis(),
        executor=processor.futures_executor,
        executor_args={'schema': NanoAODSchema, 'workers': 4},
        chunksize=100000,
        maxchunks=None,
    )
    elapsed = time.time() - tstart
    nEvent = output['nEvent'][dataset]
    print(f'{nEvent/elapsed:,.0f} events/s total')


    with open(f'{output_path}/hists.pkl', 'wb') as hfile:
        pickle.dump(output, hfile)

    with open(f'{output_path}/hists.pkl', 'rb') as hfile:
        hists = pickle.load(hfile)
        ax = hist.plot1d(output['hists']['passPreSel']['fourTag']['SR']['SvB_ps_hh'], overlay='trigWeight')
        fig = ax.get_figure()
        fig.savefig('test.pdf')
        # fig.savefig('~/nobackup/ZZ4b/uproot_plots/2016/HH4b/fourTag/SR/SvB_ps')







    # metadata = {}
    # metadata['fpath'] = path
    # metadata['fname'] = path+event_file
    # metadata['sample'] = 'HH'
    # metadata['year'] = 2016
    # metadata['lumi'] = 36.3e3
    # metadata['xs'] = 0.03105*0.5824**2
    # metadata['kFactor'] = 1
    # metadata['isMC'] = True
    # with uproot.open(path+event_file) as rfile:
    #     Runs = rfile['Runs']
    #     metadata['genEventSumw'] = np.sum(Runs['genEventSumw'])

    # event  = NanoEventsFactory.from_root(path+event_file,  schemaclass=NanoAODSchema, metadata=metadata).events()
    # SvB    = NanoEventsFactory.from_root(path+SvB_file,    schemaclass=MultiClassifierSchema).events().SvB
    # SvB_MA = NanoEventsFactory.from_root(path+SvB_MA_file, schemaclass=MultiClassifierSchema).events().SvB_MA
    # event['SvB']    = SvB
    # event['SvB_MA'] = SvB_MA
    # nEvent = len(event)

    # a = analysis()
    # output = a.process(event)
