# unset PYTHONPATH
# source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc8-opt/setup.sh 
import pickle, os, time
from copy import deepcopy
from dataclasses import dataclass
import awkward as ak
import numpy as np
import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
from coffea import processor, hist, util
# from hist import Hist # https://hist.readthedocs.io/en/latest/
# import hist
#from coffea.btag_tools import BTagScaleFactor
import correctionlib
import correctionlib._core as core
# from coffea.lookup_tools import extractor
# from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
# from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
# from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
import cachetools

from MultiClassifierSchema import MultiClassifierSchema
from functools import partial
from multiprocessing import Pool

import torch
import torch.nn.functional as F
from networks import HCREnsemble
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# print(torch.__config__.parallel_info())



@dataclass
class variable:
    def __init__(self, name, bins, label='Events'):
        self.name = name
        self.bins = bins
        self.label = label


def fourvectorhists(name, title, pt=(100, 0, 500), mass=(100, 0, 500), label='Events', extras=[]):
    variables = []
    variables += [variable(f'{name}.pt',   hist.Bin('x', f'{title} p$_{{\\mathrm{{T}}}}$ [GeV]', pt[0], pt[1], pt[2]), label)]
    variables += [variable(f'{name}.eta',  hist.Bin('x', f'{title} $\\eta$', 100, -5, 5), label)]
    variables += [variable(f'{name}.phi',  hist.Bin('x', f'{title} $\\phi$',  60, -np.pi, np.pi), label)]
    variables += [variable(f'{name}.mass', hist.Bin('x', f'{title} Mass [GeV]',  mass[0], mass[1], mass[2]), label)]

    variables += [variable(f'{name}.pz',     hist.Bin('x', f'{title} p$_{{\\mathrm{{z}}}}$ [GeV]', 150, 0, 1500), label)]
    variables += [variable(f'{name}.energy', hist.Bin('x', f'{title} Energy [GeV]', 100, 0, 500), label)]

    for var in extras:
        bins = deepcopy(var.bins)
        bins.label = f'{title} {var.bins.label}'
        variables += [variable(f'{name}.{var.name}', bins, label)]

    return variables


class jetCombinatoricModel:
    def __init__(self, filename, cut='passPreSel'):
        self.filename = filename
        self.cut = cut
        self.read_parameter_file()
        # print(self.data)

    def read_parameter_file(self):
        self.data = {}
        with open(self.filename, 'r') as lines:
            for line in lines:
                words = line.split()
                if not len(words): continue
                if len(words) == 2:
                    self.data[words[0]] = float(words[1])
                else:
                    self.data[words[0]] = ' '.join(words[1:])

        self.p = self.data[f'pseudoTagProb_{self.cut}'] 
        self.e = self.data[f'pairEnhancement_{self.cut}']
        self.d = self.data[f'pairEnhancementDecay_{self.cut}']
        self.t = self.data[f'threeTightTagFraction_{self.cut}']

    def __call__(self, untagged_jets):
        nEvent = len(untagged_jets)
        maxPseudoTags = 12
        nbt = 3 # number of required b-tags
        nlt = ak.to_numpy( ak.num(untagged_jets, axis=1) ) # number of light jets
        nPseudoTagProb = np.zeros((maxPseudoTags+1, nEvent))
        nPseudoTagProb[0] = self.t * (1-self.p)**nlt
        for npt in range(1,maxPseudoTags+1): # iterate over all possible number of pseudo-tags
            nt  = nbt + npt # number of tagged jets (b-tagged or pseudo-tagged)
            nnt = nlt - npt # number of not tagged jets (b-tagged or pseudo-tagged)
            nnt[nnt<0] = 0 # in cases where npt>nlt, set nnt to zero
            ncr = ak.to_numpy( ak.num(ak.combinations(untagged_jets, npt)) ) # number of ways to get npt pseudo-tags
            w_npt = self.t * ncr * self.p**npt * (1-self.p)**nnt
            if (nt%2)==0: # event number of tags boost from pair production enhancement term
                w_npt *= 1 + self.e/nlt**self.d
            nPseudoTagProb[npt] = w_npt
        w = np.sum(nPseudoTagProb[1:], axis=0)
        r = np.random.uniform(0,1, size=nEvent)*w + nPseudoTagProb[0] # random number between nPseudoTagProb[0] and nPseudoTagProb.sum(axis=0)
        r = r.reshape(1,nEvent).repeat(maxPseudoTags+1,0)
        c = np.array([nPseudoTagProb[:npt+1].sum(axis=0) for npt in range(maxPseudoTags+1)]) # cumulative prob        
        npt = (r>c).sum(axis=0)
        return w, npt
        

def juncWS_file(era='UL18', condor=False):
    # APV == preVFP
    # got .tar.gz of weight sets from twiki: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC
    # Only the UncertaintySources files are needed to get the JES variations
    weight_sets = {'UL16_preVFP' : ['* * nTupleAnalysis/baseClasses/data/Summer19UL16APV_V7_MC/RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   'UL16_postVFP': ['* * nTupleAnalysis/baseClasses/data/Summer19UL16_V7_MC/RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   'UL17'        : ['* * nTupleAnalysis/baseClasses/data/Summer19UL17_V5_MC/RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   'UL18'        : ['* * nTupleAnalysis/baseClasses/data/Summer19UL18_V5_MC/RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   '2016'        : ['* * nTupleAnalysis/baseClasses/data/Summer16_07Aug2017_V11_MC/RegroupedV2_Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   '2017'        : ['* * nTupleAnalysis/baseClasses/data/Fall17_17Nov2017_V32_MC/RegroupedV2_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt'],
                   '2018'        : ['* * nTupleAnalysis/baseClasses/data/Autumn18_V19_MC/RegroupedV2_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt'],
               }
    if condor:
        weight_sets['UL16_preVFP']  = ['* * RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['UL16_postVFP'] = ['* * RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['UL17']         = ['* * RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['UL18']         = ['* * RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['2016']         = ['* * RegroupedV2_Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['2017']         = ['* * RegroupedV2_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt']
        weight_sets['2018']         = ['* * RegroupedV2_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt']
    return weight_sets[era]

# following example here: https://github.com/CoffeaTeam/coffea/blob/master/tests/test_jetmet_tools.py#L529
def init_jet_factory(weight_sets, debug=False):
    from coffea.lookup_tools import extractor
    extract = extractor()
    extract.add_weight_sets(weight_sets)
    extract.finalize()
    evaluator = extract.make_evaluator()

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
    jec_stack_names = []
    for key in evaluator.keys():
        # print(key)
        if 'UncertaintySources' in key:
            jec_stack_names.append(key)

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    # print('jec_inputs:')
    # print(jec_inputs)
    jec_stack = JECStack(jec_inputs)
    # print('jec_stack')
    # print(jec_stack.__dict__)
    name_map = jec_stack.blank_name_map
    name_map["JetPt"]    = "pt"
    name_map["JetMass"]  = "mass"
    name_map["JetEta"]   = "eta"
    name_map["JetA"]     = "area"
    # name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw']    = 'pt_raw'
    name_map['massRaw']  = 'mass_raw'
    name_map['Rho']      = 'rho'
    # print(name_map)

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    uncertainties = jet_factory.uncertainties()
    if uncertainties:
        if debug: 
            for unc in uncertainties: 
                print(unc)
    else:
        print('WARNING: No uncertainties were loaded in the jet factory')

    return jet_factory


def count_nested_dict(nested_dict, c=0):
    for key in nested_dict:
        if isinstance(nested_dict[key], dict):
            c = count_nested_dict(nested_dict[key], c)
        else:
            c += 1
    return c

class analysis(processor.ProcessorABC):
    def __init__(self, debug = False, JCM = '', btagVariations=None, juncVariations=None, SvB=None, threeTag = True):
        self.debug = debug
        self.blind = True
        print('Initialize Analysis Processor')
        self.cuts = ['passPreSel']
        self.threeTag = threeTag
        self.tags = ['threeTag','fourTag'] if threeTag else ['fourTag']
        self.regions = ['inclusive','SBSR','SB','SR']
        self.regions = ['SR']
        self.signals = ['zz','zh','hh']
        self.JCM = jetCombinatoricModel(JCM)
        self.btagVar = btagVariations
        self.juncVar = juncVariations
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None


        self.variables = []
        self.variables += [variable(f'SvB_ps_{bb}', hist.Bin('x', f'SvB Regressed P(Signal) $|$ P({bb.upper()}) is largest', 100, 0, 1)) for bb in self.signals]
        self.variables += [variable(f'SvB_ps_all',  hist.Bin('x', f'SvB Regressed P(Signal)',                                100, 0, 1))]
        self.variables += [variable(f'SvB_MA_ps_{bb}', hist.Bin('x', f'SvB MA Regressed P(Signal) $|$ P({bb.upper()}) is largest', 100, 0, 1)) for bb in self.signals]
        self.variables += [variable(f'SvB_MA_ps_all',  hist.Bin('x', f'SvB MA Regressed P(Signal)',                                100, 0, 1))]
        self.variables_systematics = self.variables[0:8]
        self.variables += [variable('nJet_selected', hist.Bin('x', 'Number of Selected Jets', 16, -0.5, 15.5))]
        self.variables += fourvectorhists('canJet', 'Boson Candidate Jets', mass=(50, 0, 50), label='Jets')
        self.variables += fourvectorhists('v4Jet', 'Diboson Candidate', mass=(120, 0, 1200))

        diJet_extras = [variable('dr', hist.Bin('x', '$\\Delta$R(j,j)', 50, 0, 5)),
                        variable('st', hist.Bin('x', 's$_{{\\mathrm{{T}}}}$ [GeV]', 100, 0, 1000))]
        self.variables += fourvectorhists('quadJet_selected.lead', 'Leading Boson Candidate', extras=diJet_extras)
        self.variables += fourvectorhists('quadJet_selected.subl', 'Subleading Boson Candidate', extras=diJet_extras)
        self.variables += [variable('quadJet_selected.dr', hist.Bin('x', 'Selected Diboson Candidate $\\Delta$R(d,d)', 50, 0, 5))]
        self.variables += [variable(f'quadJet_selected.x{bb.upper()}', hist.Bin('x', f'Selected Diboson Candidate X$_{{\\mathrm{bb.upper()}}}$', 100, 0, 10)) for bb in self.signals]
        
        self._accumulator = processor.dict_accumulator({'cutflow': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, partial(processor.defaultdict_accumulator, float) ) ),
                                                        'nEvent' : processor.defaultdict_accumulator(int),
                                                        'hists'  : processor.dict_accumulator()})

        for junc in self.juncVar:
            print(f'Making hists for {junc}')
            self._accumulator['hists'][junc] = processor.dict_accumulator()
            for cut in self.cuts:
                print(f'    {cut}')
                self._accumulator['hists'][junc][cut] = processor.dict_accumulator()
                for tag in self.tags:
                    print(f'        {tag}')
                    self._accumulator['hists'][junc][cut][tag] = processor.dict_accumulator()
                    for region in self.regions:
                        print(f'            {region}')
                        self._accumulator['hists'][junc][cut][tag][region] = processor.dict_accumulator()
                        for var in self.variables:
                            self._accumulator['hists'][junc][cut][tag][region][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)

            self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_Bool'] = processor.dict_accumulator()
            self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_Data'] = processor.dict_accumulator()
            self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_MC']   = processor.dict_accumulator()
            for syst in self.btagVar: 
                self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR'][f'btagSF_{syst}'] = processor.dict_accumulator()
            for syst in self.juncVar:
                self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR'][       f'{syst}'] = processor.dict_accumulator()

            for var in self.variables_systematics:
                self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_Bool'][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)
                self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_Data'][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)
                self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR']['trigWeight_MC'  ][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)
                for syst in self.btagVar:
                    self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR'][f'btagSF_{syst}'][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)
                # for syst in self.juncVar:
                #     self._accumulator['hists'][junc]['passPreSel']['fourTag']['SR'][       f'{syst}'][var.name] = hist.Hist(var.label, hist.Cat('dataset', 'Dataset'), var.bins)

        self.nHists = count_nested_dict(self._accumulator['hists'])
        print(f'{self.nHists} total histograms')
        
    @property
    def accumulator(self):
        return self._accumulator

    def cutflow(self, output, event, cut, allTag=False, junc='JES_Central'):
        if allTag:
            w = event.weight
            sumw = np.sum(w)
            sumw_3, sumw_4 = sumw, sumw
        else:
            e3, e4 = event[event.threeTag], event[event.fourTag]
            sumw_3 = np.sum(e3.weight)
            sumw_4 = np.sum(e4.weight)

        output['cutflow'][junc]['threeTag'][cut] += sumw_3
        output['cutflow'][junc][ 'fourTag'][cut] += sumw_4

        if event.metadata.get('isMC', False) and not allTag:
            output['cutflow'][junc]['threeTag'][cut+'_HLT_Bool'] += np.sum(e3.weight*e3.passHLT)
            output['cutflow'][junc][ 'fourTag'][cut+'_HLT_Bool'] += np.sum(e4.weight*e4.passHLT)

            output['cutflow'][junc]['threeTag'][cut+'_HLT_MC'  ] += np.sum(e3.weight*e3.trigWeight.MC)
            output['cutflow'][junc][ 'fourTag'][cut+'_HLT_MC'  ] += np.sum(e4.weight*e4.trigWeight.MC)

            output['cutflow'][junc]['threeTag'][cut+'_HLT_Data'] += np.sum(e3.weight*e3.trigWeight.Data)
            output['cutflow'][junc][ 'fourTag'][cut+'_HLT_Data'] += np.sum(e4.weight*e4.trigWeight.Data)
            

    def process(self, event):
        tstart = time.time()
        output = self.accumulator.identity()

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        isMC    = event.metadata.get('isMC',  False)
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        btagSF  = event.metadata.get('btagSF', None)
        juncWS  = event.metadata.get('juncWS', None)
        nEvent = len(event)
        np.random.seed(0)
        output['nEvent'][dataset] += nEvent

        if isMC:
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            if btagSF is not None:
                btagSF = correctionlib.CorrectionSet.from_file(btagSF)['deepJet_shape']

        if self.debug: print(fname)
        if self.debug: print(f'{chunk}Process {nEvent} Events')

        path = fname.replace(fname.split('/')[-1],'')
        event['SvB']    = NanoEventsFactory.from_root(f'{path}SvB.root',    entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB
        event['SvB_MA'] = NanoEventsFactory.from_root(f'{path}SvB_MA.root', entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB_MA

        event['SvB', 'passMinPs'] = (event.SvB.pzz>0.01) | (event.SvB.pzh>0.01) | (event.SvB.phh>0.01) 
        event['SvB', 'zz'] = (event.SvB.pzz >  event.SvB.pzh) & (event.SvB.pzz >  event.SvB.phh)
        event['SvB', 'zh'] = (event.SvB.pzh >  event.SvB.pzz) & (event.SvB.pzh >  event.SvB.phh)
        event['SvB', 'hh'] = (event.SvB.phh >= event.SvB.pzz) & (event.SvB.phh >= event.SvB.pzh)

        event['SvB_MA', 'passMinPs'] = (event.SvB_MA.pzz>0.01) | (event.SvB_MA.pzh>0.01) | (event.SvB_MA.phh>0.01) 
        event['SvB_MA', 'zz'] = (event.SvB_MA.pzz >  event.SvB_MA.pzh) & (event.SvB_MA.pzz >  event.SvB_MA.phh)
        event['SvB_MA', 'zh'] = (event.SvB_MA.pzh >  event.SvB_MA.pzz) & (event.SvB_MA.pzh >  event.SvB_MA.phh)
        event['SvB_MA', 'hh'] = (event.SvB_MA.phh >= event.SvB_MA.pzz) & (event.SvB_MA.phh >= event.SvB_MA.pzh)

        if isMC:
            for junc in self.juncVar:
                output['cutflow'][junc]['threeTag']['all'] = lumi * xs * kFactor    
                output['cutflow'][junc][ 'fourTag']['all'] = lumi * xs * kFactor    
        else:
            self.cutflow(output, event, 'all', allTag = True)


        # Get trigger decisions 
        if year == '2016':
            event['passHLT'] = event.HLT.QuadJet45_TripleBTagCSV_p087 | event.HLT.DoubleJet90_Double30_TripleBTagCSV_p087 | event.HLT.DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6
        if year == '2017':
            event['passHLT'] = event.HLT.PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0 | event.HLT.DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33
        if year == '2018':
            event['passHLT'] = event.HLT.DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 | event.HLT.PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5

        if not isMC: # for data, apply trigger cut first thing, for MC, keep all events and apply trigger in cutflow and for plotting
            event = event[event.passHLT]

        if isMC:
            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)


        #
        # Calculate and apply Jet Energy Calibration
        #
        if isMC and juncWS is not None:
            jet_factory = init_jet_factory(juncWS)

            nominal_jet = event.Jet
            nominal_jet['pt_raw']   = (1 - nominal_jet.rawFactor) * nominal_jet.pt
            nominal_jet['mass_raw'] = (1 - nominal_jet.rawFactor) * nominal_jet.mass
            # nominal_jet['pt_gen']   = ak.values_astype(ak.fill_none(nominal_jet.matched_gen.pt, 0), np.float32)
            nominal_jet['rho']      = ak.broadcast_arrays(event.fixedGridRhoFastjetAll, nominal_jet.pt)[0]

            jec_cache = cachetools.Cache(np.inf)
            jet_variations = jet_factory.build(nominal_jet, lazy_cache=jec_cache)

            
        #
        # Loop over jet energy uncertainty variations running event selection, filling hists/cuflows independently for each jet calibration
        #
        for junc in self.juncVar:
            if junc != 'JES_Central':
                if self.debug: print(f'{chunk} running selection for {junc}')
                variation = '_'.join(junc.split('_')[:-1]).replace('YEAR', year)
                direction = junc.split('_')[-1]
                event['Jet'] = jet_variations[variation, direction]

            event['Jet' ,'selected'] = (event.Jet.pt>=40) & (np.abs(event.Jet.eta)<=2.4) & ~((event.Jet.puId<0b110)&(event.Jet.pt<50))
            event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

            selev = event[event.nJet_selected >= 4]
            self.cutflow(output, selev, 'passJetMult', allTag = True, junc=junc)


            #
            # Build and select boson candidate jets with bRegCorr applied
            # 
            selev['selJet'] = selev.Jet[selev.Jet.selected]
            canJet = selev.selJet * selev.selJet.bRegCorr
            canJet['btagDeepFlavB'] = selev.selJet.btagDeepFlavB
            if isMC:
                canJet['hadronFlavour'] = selev.selJet.hadronFlavour

            # sort by btag score
            canJet = canJet[ak.argsort(canJet.btagDeepFlavB, axis=1, ascending=False)]
            # take top four
            canJet = canJet[:,0:4]
            # return to pt sorting 
            canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
            selev['canJet'] = canJet
            selev['v4j'] = canJet.sum(axis=1)

            #
            # Calculate and apply btag scale factors
            #
            if isMC and btagSF is not None:
                #central = 'central'
                use_central = True
                btag_jes = []
                if junc != 'JES_Central' and 'JES_Total' not in junc:
                    use_central = False
                    btag_jes = [f'{direction}_jes{variation.replace("JES_","")}']
                cj, nj = ak.flatten(selev.canJet), ak.num(selev.canJet)
                hf, eta, pt, tag = np.array(cj.hadronFlavour), np.array(abs(cj.eta)), np.array(cj.pt), np.array(cj.btagDeepFlavB)

                cj_bl = selev.canJet[selev.canJet.hadronFlavour!=4]
                nj_bl = ak.num(cj_bl)
                cj_bl = ak.flatten(cj_bl)
                hf_bl, eta_bl, pt_bl, tag_bl = np.array(cj_bl.hadronFlavour), np.array(abs(cj_bl.eta)), np.array(cj_bl.pt), np.array(cj_bl.btagDeepFlavB)
                SF_bl= btagSF.evaluate('central', hf_bl, eta_bl, pt_bl, tag_bl)
                SF_bl = ak.unflatten(SF_bl, nj_bl)
                SF_bl = np.prod(SF_bl, axis=1)

                cj_c = selev.canJet[selev.canJet.hadronFlavour==4]
                nj_c = ak.num(cj_c)
                cj_c = ak.flatten(cj_c)
                hf_c, eta_c, pt_c, tag_c = np.array(cj_c.hadronFlavour), np.array(abs(cj_c.eta)), np.array(cj_c.pt), np.array(cj_c.btagDeepFlavB)
                SF_c= btagSF.evaluate('central', hf_c, eta_c, pt_c, tag_c)
                SF_c = ak.unflatten(SF_c, nj_c)                    
                SF_c = np.prod(SF_c, axis=1)

                for sf in self.btagVar+btag_jes:
                    if sf == 'central':
                        SF = btagSF.evaluate('central', hf, eta, pt, tag)
                        SF = ak.unflatten(SF, nj)
                        SF = np.prod(SF, axis=1)
                    if '_cf' in sf:
                        SF = btagSF.evaluate(sf, hf_c, eta_c, pt_c, tag_c)
                        SF = ak.unflatten(SF, nj_c)                    
                        SF = SF_bl * np.prod(SF, axis=1) # use central value for b,l jets
                    if '_hf' in sf or '_lf' in sf or '_jes' in sf:
                        SF = btagSF.evaluate(sf, hf_bl, eta_bl, pt_bl, tag_bl)
                        SF = ak.unflatten(SF, nj_bl)
                        SF = SF_c * np.prod(SF, axis=1) # use central value for charm jets

                    selev[f'btagSF_{sf}'] = SF 
                    selev[f'weight_btagSF_{sf}'] = selev.weight * SF

                selev.weight = selev[f'weight_btagSF_{"central" if use_central else btag_jes[0]}']
                self.cutflow(output, selev, 'passJetMult_btagSF', allTag = True, junc=junc)


            selev['Jet', 'tagged']       = selev.Jet.selected & (selev.Jet.btagDeepFlavB>=0.6)
            selev['Jet', 'tagged_loose'] = selev.Jet.selected & (selev.Jet.btagDeepFlavB>=0.3)
            selev['nJet_tagged']         = ak.num(selev.Jet[selev.Jet.tagged])
            selev['nJet_tagged_loose']   = ak.num(selev.Jet[selev.Jet.tagged_loose])


            fourTag  = (selev['nJet_tagged']       >= 4)
            threeTag = (selev['nJet_tagged_loose'] == 3) & (selev['nJet_selected'] >= 4)
            # check that coffea jet selection agrees with c++
            if junc == 'JES_Central':
                selev['issue'] = (threeTag!=selev.threeTag)|(fourTag!=selev.fourTag)
                if ak.any(selev.issue):
                    print(f'{chunk}WARNING: selected jets or fourtag calc not equal to picoAOD values')
                    print('nSelJets')
                    print(selev[selev.issue].nSelJets)
                    print(selev[selev.issue].nJet_selected)
                    print('fourTag')
                    print(selev.fourTag[selev.issue])
                    print(fourTag[selev.issue])

            selev[ 'fourTag']   =  fourTag
            selev['threeTag']   = threeTag * self.threeTag
            selev['passPreSel'] = threeTag | fourTag

            #
            # Preselection: keep only three or four tag events
            # 
            selev = selev[selev['threeTag'] | selev['fourTag']]

            if self.threeTag:
                #
                # calculate pseudoTagWeight for threeTag events
                #
                selev['Jet_untagged_loose'] = selev.Jet[selev.Jet.selected & ~selev.Jet.tagged_loose]
                nJet_pseudotagged = np.zeros(len(selev), dtype=int)
                pseudoTagWeight = np.ones(len(selev))
                pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = self.JCM(selev[selev.threeTag]['Jet_untagged_loose'])
                selev['nJet_pseudotagged'] = nJet_pseudotagged

                # check that pseudoTagWeight calculation agrees with c++
                if junc == 'JES_Central':
                    selev.issue = (abs(selev.pseudoTagWeight - pseudoTagWeight)/selev.pseudoTagWeight > 0.0001) & (selev.pseudoTagWeight!=1)
                    if ak.any(selev.issue):
                        print(f'{chunk}WARNING: python pseudotag calc not equal to c++ calc')
                        print(f'{chunk}Issues:',ak.sum(selev.issue),'of',ak.sum(selev.threeTag))

                # add pseudoTagWeight to event
                selev['pseudoTagWeight'] = pseudoTagWeight

                # apply pseudoTagWeight to threeTag events
                e3 = selev[selev.threeTag]
                selev[selev.threeTag].weight = e3.weight * e3.pseudoTagWeight

            # presel cutflow with pseudotag weight included
            self.cutflow(output, selev, 'passPreSel', junc=junc)


            #
            # Build diJets, indexed by diJet[event,pairing,0/1]
            #
            canJet = selev['canJet']
            pairing = [([0,2],[0,1],[0,1]),
                       ([1,3],[2,3],[3,2])]
            diJet       = canJet[:,pairing[0]]     +   canJet[:,pairing[1]]
            diJet['st'] = canJet[:,pairing[0]].pt  +   canJet[:,pairing[1]].pt
            diJet['dr'] = canJet[:,pairing[0]].delta_r(canJet[:,pairing[1]])
            diJet['lead'] = canJet[:,pairing[0]]
            diJet['subl'] = canJet[:,pairing[1]]
            # Sort diJets within views to be lead st, subl st
            diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
            # Now indexed by diJet[event,pairing,lead/subl st]

            # Compute diJetMass cut with independent min/max for lead/subl 
            minDiJetMass = np.array([[[ 52, 50]]])
            maxDiJetMass = np.array([[[180,173]]])
            diJet['passDiJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

            # Compute MDRs
            min_m4j_scale = np.array([[ 360, 235]])
            min_dr_offset = np.array([[-0.5, 0.0]])
            max_m4j_scale = np.array([[ 650, 650]])
            max_dr_offset = np.array([[ 0.5, 0.7]])
            max_dr        = np.array([[ 1.5, 1.5]])
            m4j = np.repeat(np.reshape(np.array(selev['v4j'].mass), (-1,1,1)), 2, axis=2)
            diJet['passMDR'] = (min_m4j_scale/m4j + min_dr_offset < diJet.dr) & (diJet.dr < np.maximum(max_m4j_scale/m4j + max_dr_offset, max_dr))

            # Compute consistency of diJet masses with boson masses
            mZ =  91.0
            mH = 125.0
            st_bias = np.array([[[1.02, 0.98]]])
            cZ = mZ * st_bias
            cH = mH * st_bias

            diJet['xZ'] = (diJet.mass - cZ)/(0.1*diJet.mass)
            diJet['xH'] = (diJet.mass - cH)/(0.1*diJet.mass)


            #
            # Build quadJets
            #
            quadJet = ak.zip({'lead': diJet[:,:,0],
                              'subl': diJet[:,:,1],
                              'passDiJetMass': ak.all(diJet.passDiJetMass, axis=2),
                              'random': np.random.uniform(low=0.1, high=0.9, size=(diJet.__len__(), 3))
                          })#, with_name='quadJet')
            quadJet['dr'] = quadJet['lead'].delta_r(quadJet['subl'])
            quadJet['SvB_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB.q_1234), (-1,1)), 
                                                     np.reshape(np.array(selev.SvB.q_1324), (-1,1)),
                                                     np.reshape(np.array(selev.SvB.q_1423), (-1,1))), axis=1)
            quadJet['SvB_MA_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB_MA.q_1234), (-1,1)), 
                                                        np.reshape(np.array(selev.SvB_MA.q_1324), (-1,1)),
                                                        np.reshape(np.array(selev.SvB_MA.q_1423), (-1,1))), axis=1)

            # Compute Signal Regions
            quadJet['xZZ'] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
            quadJet['xHH'] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
            quadJet['xZH'] = np.sqrt(np.minimum(quadJet.lead.xH**2 + quadJet.subl.xZ**2, 
                                                quadJet.lead.xZ**2 + quadJet.subl.xH**2))
            max_xZZ = 2.6
            max_xZH = 1.9
            max_xHH = 1.9
            quadJet['ZZSR'] = quadJet.xZZ < max_xZZ
            quadJet['ZHSR'] = quadJet.xZH < max_xZH
            quadJet['HHSR'] = quadJet.xHH < max_xHH
            quadJet['SR'] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
            quadJet['SB'] = quadJet.passDiJetMass & ~quadJet.SR

            # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
            quadJet['rank'] = 10*quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random
            quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

            selev[  'diJet'] =   diJet
            selev['quadJet'] = quadJet
            selev['quadJet_selected'] = quadJet[quadJet.selected][:,0]

            # selev.issue = (selev.leadStM<0) | (selev.sublStM<0)
            # if ak.any(selev.issue):
            #     print(f'{chunk}WARNING: Negative diJet masses in picoAOD variables generated by the c++')
            #     issue = selev[selev.issue]
            #     print(f'{chunk}{len(issue)} events with issues')
            #     print(f'{chunk}c++ values:',issue.passDiJetMass, issue.leadStM,issue.sublStM)
            #     print(f'{chunk}py  values:',issue.quadJet_selected.passDiJetMass, issue.quadJet_selected.lead.mass, issue.quadJet_selected.subl.mass)            

            if junc == 'JES_Central':
                selev.issue = selev.passDijetMass != selev['quadJet_selected'].passDiJetMass
                selev.issue = selev.issue & ~((selev.leadStM<0) | (selev.sublStM<0))
                if ak.any(selev.issue):
                    print(f'{chunk}WARNING: passDiJetMass calc not equal to picoAOD value')
                    issue = selev[selev.issue]
                    print(f'{chunk}{len(issue)} events with issues')
                    print(f'{chunk}c++ values:',issue.passDijetMass, issue.leadStM,issue.sublStM)
                    print(f'{chunk}py  values:',issue.quadJet_selected.passDiJetMass, issue.quadJet_selected.lead.mass, issue.quadJet_selected.subl.mass)            


            # Blind data in fourTag SR
            if not isMC and self.blind:
                selev = selev[~(selev.SR & selev.fourTag)]

            self.cutflow(output, selev[selev['quadJet_selected'].passDiJetMass], 'passDiJetMass', junc=junc)
            self.cutflow(output, selev[selev['quadJet_selected'].SR], 'SR', junc=junc)

            if self.classifier_SvB is not None:
                self.compute_SvB(selev, junc=junc)

            #
            # fill histograms
            #
            self.fill(selev, output, junc=junc)

            if isMC:
                self.fill_systematics(selev, output, junc=junc)



        # Done
        elapsed = time.time() - tstart
        if self.debug: print(f'{chunk}{nEvent/elapsed:,.0f} events/s')
        return output


    def compute_SvB(self, event, junc='JES_Central'):
        n = len(event)

        j = torch.zeros(n, 4, 4)
        j[:,0,:] = torch.tensor( event['canJet'].pt   )
        j[:,1,:] = torch.tensor( event['canJet'].eta  )
        j[:,2,:] = torch.tensor( event['canJet'].phi  )
        j[:,3,:] = torch.tensor( event['canJet'].mass )

        o = torch.zeros(n, 5, 8)

        a = torch.zeros(n, 4)
        a[:,0] =        float( event.metadata['year'][3] )
        a[:,1] = torch.tensor( event['nJet_selected'] )
        a[:,2] = torch.tensor( event.xW )
        a[:,3] = torch.tensor( event.xbW )

        e = torch.tensor(event.event)%3

        c_logits, q_logits = self.classifier_SvB(j, o, a, e)

        c_score, q_score = F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()

        # classes = [mj,tt,zz,zh,hh]
        SvB = ak.zip({'pmj': c_score[:,0],
                      'ptt': c_score[:,1],
                      'pzz': c_score[:,2],
                      'pzh': c_score[:,3],
                      'phh': c_score[:,4],
                  })
        SvB['ps'] = SvB.pzz + SvB.pzh + SvB.phh
        SvB['passMinPs'] = (SvB.pzz>0.01) | (SvB.pzh>0.01) | (SvB.phh>0.01) 
        SvB['zz'] = (SvB.pzz >  SvB.pzh) & (SvB.pzz >  SvB.phh)
        SvB['zh'] = (SvB.pzh >  SvB.pzz) & (SvB.pzh >  SvB.phh)
        SvB['hh'] = (SvB.phh >= SvB.pzz) & (SvB.phh >= SvB.pzh)

        if junc == 'JES_Central':
            error = ~np.isclose(event.SvB.ps, SvB.ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                print('Calculated SvB does not agree within tolerance for some events:',np.sum(error), event.SvB.ps[error] - SvB.ps[error])
        
        event['SvB'] = SvB


    def fill_SvB(self, hist, event, weight):
        dataset = event.metadata.get('dataset','')
        for classifier in ['SvB', 'SvB_MA']:
            for bb in self.signals:
                mask = event[classifier][bb]
                x, w = event[mask][classifier].ps, weight[mask]
                hist[f'{classifier}_ps_{bb}'].fill(dataset=dataset, x=x, weight=w)
                hist[f'{classifier}_ps_{bb}'].fill(dataset=dataset, x=x, weight=w)
                hist[f'{classifier}_ps_{bb}'].fill(dataset=dataset, x=x, weight=w)

            mask = event[classifier]['zz'] | event[classifier]['zh'] | event[classifier]['hh']
            x, w = event[mask][classifier].ps, weight[mask]
            hist[f'{classifier}_ps_all'].fill(dataset=dataset, x=x, weight=w)
            hist[f'{classifier}_ps_all'].fill(dataset=dataset, x=x, weight=w)
            hist[f'{classifier}_ps_all'].fill(dataset=dataset, x=x, weight=w)
                

    def fill_fourvectorhists(self, name, hist, event, weight):
        dataset = event.metadata.get('dataset','')
        namepath = tuple(name.split('.'))

        _, w = ak.broadcast_arrays(event[namepath].pt, weight, depth_limit=1) # duplicate event weights so that we can fill with multiple objects per event
        try:
            v, w = ak.flatten(event[namepath]), ak.flatten(w) # flatten arrays for filling, allows for multiple objects per event
        except ValueError:
            v = event[namepath] # cannot flatten arrays of records. Hopefully won't need to deal with multiple records per event...

        for var in ['pt','eta','phi','mass', 'pz','energy','dr','st']:
            try:
                hist[f'{name}.{var}'].fill(dataset=dataset, x=getattr(v,var), weight=w)
            except KeyError:
                pass # histogram for this attribute was not initialized


    def fill(self, event, output, junc='JES_Central'):
        dataset = event.metadata.get('dataset','')
        isMC    = event.metadata.get('isMC', False)
        for cut in self.cuts:
            for tag in self.tags:
                mask_cut_tag = event[tag] & event[cut]
                for region in self.regions:
                    if   region == 'SBSR':
                        mask = mask_cut_tag & (event['quadJet_selected'].SB | event['quadJet_selected'].SR)
                    elif region == 'SB':
                        mask = mask_cut_tag & event['quadJet_selected'].SB
                    elif region == 'SR':
                        mask = mask_cut_tag & event['quadJet_selected'].SR
                    elif region == 'inclusive':
                        mask = mask_cut_tag

                    hist_event = event[mask]
                    weight = hist_event.weight
                    if isMC: weight = weight * hist_event.trigWeight.Data

                    hist = output['hists'][junc][cut][tag][region]
                    hist['nJet_selected'].fill(dataset=dataset, x=hist_event.nJet_selected, weight=weight)
                    #hist['canJet_pt'].fill(dataset=dataset, x=hist_event.canJet.pt, weight=weight)
                    self.fill_fourvectorhists('canJet', hist, hist_event, weight)
                    self.fill_fourvectorhists('v4j', hist, hist_event, weight)
                    self.fill_fourvectorhists('quadJet_selected.lead', hist, hist_event, weight)
                    self.fill_fourvectorhists('quadJet_selected.subl', hist, hist_event, weight)
                    hist['quadJet_selected.dr'].fill(dataset=dataset, x=hist_event['quadJet_selected'].dr, weight=weight)
                    for bb in self.signals: hist[f'quadJet_selected.x{bb.upper()}'].fill(dataset=dataset, x=hist_event['quadJet_selected', f'x{bb.upper()}'], weight=weight)
                    self.fill_SvB(hist, hist_event, weight)

        
    def fill_systematics(self, event, output, junc='JES_Central'):
        mask = event['fourTag']
        mask = mask & event['quadJet_selected'].SR
        event = event[mask]

        #def fill_SvB(self, hist, event, weight):
        for trig in ['Bool', 'MC', 'Data']:
            hist = output['hists'][junc]['passPreSel']['fourTag']['SR'][f'trigWeight_{trig}']
            weight = event.weight * event.passHLT if trig == 'Bool' else event.weight * event.trigWeight[trig]
            self.fill_SvB(hist, event, weight)

        for sf in self.btagVar:
            hist = output['hists'][junc]['passPreSel']['fourTag']['SR'][f'btagSF_{sf}']
            weight = event[f'weight_btagSF_{sf}'] * event.trigWeight.Data
            self.fill_SvB(hist, event, weight)
            

    def postprocess(self, accumulator):
        return accumulator




# for MC we need to normalize the sample to the recommended cross section * BR times the target luminosity
## Higgs BRs https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR BR(h125->bb) = 0.5824 BR(h125->\tau\tau) = 0.06272 BR(Z->bb) = 0.1512, BR(Z->\tau\tau) = 0.03696
## ZH cross sections https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
## ZZ cross section 15.0 +0.7 -0.6 +/-0.2 (MCFM at NLO in QCD with additional contributions from LO gg -> ZZ diagrams) or 16.2 +0.6 -0.4 (calculated at NNLO in QCD via MATRIX) https://arxiv.org/pdf/1607.08834.pdf pg 10
## ZH->bb\tau\tau xs = (0.7612+0.1227)*(0.58*0.036+0.15*0.067) = 27/fb ~ 10x HH cross section
## HH->bb\tau\tau xs = 34*0.58*0.067*2 = 2.6/fb
## Higgs BR(mH=125.0) = 0.5824, BR(mH=125.09) = 0.5809: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
## Z BR = 0.1512+/-0.0005 from PDG
## store all process cross sections in pb. Can compute xs of sample with GenXsecAnalyzer. Example: 
## https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer
## cd genproductions/Utilities/calculateXSectionAndFilterEfficiency; ./calculateXSectionAndFilterEfficiency.sh -f ../../../ZZ_dataset.txt -c RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1 -d MINIAODSIM -n -1 
## tt xs NNLO and measurement in dilep and semilep tt+jets, tt+bb: https://cds.cern.ch/record/2684606/files/TOP-18-002-paper-v19.pdf
xsDictionary = {'ggZH4b':  0.1227*0.5824*0.1512, #0.0432 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.0432/0.69911 = 0.0618, almost exactly half the LHCXSWG value... NNLO = 2x NLO??
                  'ZH4b':  0.7612*0.5824*0.1512, #0.5540 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.5540/0.69911 = 0.7924, 4% larger than the LHCXSWG value.
              'bothZH4b': (0.1227+0.7612)*0.5824*0.1512,
                  'ZZ4b': 15.5   *0.1512*0.1512,#0.3688 from GenXsecAnalyzer gives 16.13 dividing by BR^2. mcEventSumw/mcEventCount * FxFx Jet Matching eff. = 542638/951791 * 0.647 = 0.3688696216. Jet matching not included in genWeight!
                  'HH4b': 0.03105*0.5824**2, # (0.0457 2018, doesn't include BR, 0.009788 2016, does include BR...) https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH recommends 31.05fb*BR^2=10.53fb
                'TTJets': 831.76, #749.5 get xs from GenXsecAnalyzer, McM is just wrong... TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8. Apply 4b scale k-factor 5.5/3.6=1.53 https://cds.cern.ch/record/2687373/files/TOP-18-011-paper-v15.pdf
                'TTToHadronic': 377.9607353256, #313.9 from McM. NNLO tt xs = 831.76, W hadronic BR = 0.6741 => NNLO = 831.76*0.6741^2 = 377.9607353256
                'TTToSemiLeptonic': 365.7826460496, #300.9 from McM. NNLO = 831.76*2*(1-0.6741)*0.6747 = 365.7826460496
                'TTTo2L2Nu': 88.3419033256, #72.1 from McM. NNLO = 831.76*(1-0.6741)^2 = 88.3419033256
                'WHHTo4B_CV_0_5_C2V_1_0_C3_1_0':2.870e-04*0.5824*0.5824,  # 2.870e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_0_0_C3_1_0':1.491e-04*0.5824*0.5824,  # 1.491e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_1_0_C3_0_0':2.371e-04*0.5824*0.5824,  # 2.371e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_1_0_C3_1_0':4.152e-04*0.5824*0.5824,  # 4.152e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_1_0_C3_2_0':6.880e-04*0.5824*0.5824,  # 6.880e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_2_0_C3_1_0':1.115e-03*0.5824*0.5824,  # 1.115e-03from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_5_C2V_1_0_C3_1_0':8.902e-04*0.5824*0.5824,  # 8.902e-04from GenXsecAnalyzer, does not include BR for H 
                'WHHTo4B_CV_1_0_C2V_1_0_C3_20_0':2.158e-02*0.5824*0.5824, # 2.158e-02from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_0_5_C2V_1_0_C3_1_0':1.663e-04*0.5824*0.5824,  # 1.663e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_0_0_C3_1_0':9.037e-05*0.5824*0.5824,  # 9.037e-05from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_1_0_C3_0_0':1.544e-04*0.5824*0.5824,  # 1.544e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_1_0_C3_1_0':2.642e-04*0.5824*0.5824,  # 2.642e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_1_0_C3_2_0':4.255e-04*0.5824*0.5824,  # 4.255e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_2_0_C3_1_0':6.770e-04*0.5824*0.5824,  # 6.770e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_5_C2V_1_0_C3_1_0':5.738e-04*0.5824*0.5824,  # 5.738e-04from GenXsecAnalyzer, does not include BR for H 
                'ZHHTo4B_CV_1_0_C2V_1_0_C3_20_0':1.229e-02*0.5824*0.5824, # 1.229e-02from GenXsecAnalyzer, does not include BR for H 
                }

lumiDict   = {'2016':  36.3e3,#35.8791
              '2016_preVFP': 19.5e3,
              '2016_postVFP': 16.5e3,
              '2017':  36.7e3,#36.7338
              '2018':  59.8e3,#59.9656
              'RunII':132.8e3,
              }

def btagSF_file(era='UL18', condor=False):
    btagSF = {'UL16_preVFP' : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz',
              'UL16_postVFP': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz',
              'UL17'        : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz',
              'UL18'        : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz',
              '2016'        : 'nTupleAnalysis/baseClasses/data/BTagSF2016/btagging_legacy16_deepJet_itFit.json.gz', # legacy for non UL HH4b sample
              '2017'        : 'nTupleAnalysis/baseClasses/data/BTagSF2017/btagging_legacy17_deepJet.json.gz',
              '2018'        : 'nTupleAnalysis/baseClasses/data/BTagSF2018/btagging_legacy18_deepJet.json.gz'}
    if condor:
        btagSF['2016'] = 'btagging_legacy16_deepJet_itFit.json.gz', # legacy for non UL HH4b sample
        btagSF['2017'] = 'btagging_legacy17_deepJet.json.gz'
        btagSF['2018'] = 'btagging_legacy18_deepJet.json.gz'

    return btagSF[era]

# def jerc_file(year='2018', UL=True, conda_pack=False):
#     jerc_UL = {'2016_preVFP' : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jet_jerc.json.gz',
#                '2016_postVFP': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jet_jerc.json.gz',
#                '2017'        : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jet_jerc.json.gz',
#                '2018'        : '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz'}

#     if UL: return jerc_UL[year]


def btagVariations(JECSyst='', systematics=False):
    btagVariations = ['central']
    if 'jes' in JECSyst:
        if 'Down' in JECSyst:
            btagVariations = ['down'+JECSyst.replace('Down','')]
        if 'Up' in JECSyst:
            btagVariations = ['up'+JECSyst.replace('Up','')]
    if systematics:
        btagVariations += ['down_hfstats1', 'up_hfstats1']
        btagVariations += ['down_hfstats2', 'up_hfstats2']
        btagVariations += ['down_lfstats1', 'up_lfstats1']
        btagVariations += ['down_lfstats2', 'up_lfstats2']
        btagVariations += ['down_hf', 'up_hf']
        btagVariations += ['down_lf', 'up_lf']
        btagVariations += ['down_cferr1', 'up_cferr1']
        btagVariations += ['down_cferr2', 'up_cferr2']
    return btagVariations


def juncVariations(systematics=False):
    juncVariations = ['JES_Central']
    if systematics:
        juncSources = ['JES_FlavorQCD',
                       'JES_RelativeBal',
                       'JES_HF',
                       'JES_BBEC1',
                       'JES_EC2',
                       'JES_Absolute',
                       'JES_Absolute_YEAR',
                       'JES_HF_YEAR',
                       'JES_EC2_YEAR',
                       'JES_RelativeSample_YEAR',
                       'JES_BBEC1_YEAR',
                       'JES_Total']
        juncVariations += [f'{juncSource}_{direction}' for juncSource in juncSources for direction in ['up', 'down']] 
    return juncVariations

if __name__ == '__main__':
    eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor'
    nfs_base = '/uscms/home/bryantp/nobackup/ZZ4b'
    eos = True

    input_path  = f'{eos_base if eos else nfs_base}'
    output_path = f'{nfs_base}'

    metadata = {}
    fileset = {}
    years = ['2016', '2017', '2018']
    #years = ['2018']
    for year in years:
        # datasets = []
        datasets = [f'HH4b{year}']
        if year == '2016':
            # datasets += ['ZZ4b2016_preVFP', 'ZZ4b2016_postVFP']
            datasets += ['ZZ4b2016_preVFP',  'ZH4b2016_preVFP',  'ggZH4b2016_preVFP']
            datasets += ['ZZ4b2016_postVFP', 'ZH4b2016_postVFP', 'ggZH4b2016_postVFP']
        else:
            # datasets += [f'ZZ4b{year}']
            datasets += [f'ZZ4b{year}', f'ZH4b{year}', f'ggZH4b{year}']
        # datasets = [f'ZZ4b{year}']
        
        for dataset in datasets:
            VFP = '_'+dataset.split('_')[-1] if 'VFP' in dataset else ''
            era = f'{20 if "HH4b" in dataset else "UL"}{year[2:]+VFP}'
            metadata[dataset] = {'isMC'  : True,
                                 'xs'    : xsDictionary[dataset.replace(year+VFP,'')],
                                 'lumi'  : lumiDict[year+VFP],
                                 'year'  : year,
                                 'btagSF': btagSF_file(era),
                                 'juncWS': juncWS_file(era),
            }
            fileset[dataset] = {'files': [f'{input_path}/{dataset}/picoAOD.root',],
                                'metadata': metadata[dataset]}

            print(f'Dataset {dataset} with {len(fileset[dataset]["files"])} files')


    analysis_args = {'debug': False,
                     'JCM': 'ZZ4b/nTupleAnalysis/weights/dataRunII/jetCombinatoricModel_SB_00-00-02.txt',
                     'btagVariations': btagVariations(systematics=False),
                     'juncVariations': juncVariations(systematics=True),
                     'threeTag': False,
                     # 'SvB': 'ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_8_np753_seed0_lr0.01_epochs20_offset*_epoch20.pkl',
    }

    tstart = time.time()
    output = processor.run_uproot_job(
        fileset,
        treename='Events',
        processor_instance=analysis(**analysis_args),
        executor=processor.futures_executor,
        executor_args={'schema': NanoAODSchema, 'workers': 1},
        chunksize=1000,
        maxchunks=1,
    )
    elapsed = time.time() - tstart
    nEvent = sum([output['nEvent'][dataset] for dataset in output['nEvent'].keys()])
    print(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')

    with open(f'{output_path}/hists.pkl', 'wb') as hfile:
        pickle.dump(output, hfile)

