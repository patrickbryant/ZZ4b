# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh
# source /cvmfs/sft.cern.ch/lcg/nightlies/dev4/Wed/coffea/0.7.13/x86_64-centos7-gcc10-opt/coffea-env.sh 
import pickle, os, time
from coffea import hist, processor



if __name__ == '__main__':
    eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor'
    nfs_base = '/uscms/home/bryantp/nobackup/ZZ4b'
    #nfs_base = os.path.expanduser(nfs_base)
    eos = True

    year = '2018'
    dataset = f'ZZ4b{year}'
    input_path = f'{eos_base if eos else nfs_base}/{dataset}'
    output_path = f'{nfs_base}/{dataset}'

    with open(f'{output_path}/hists.pkl', 'rb') as hfile:
        hists = pickle.load(hfile)
        print(hists['hists']['passPreSel']['fourTag']['SR'].keys())
        for bb in ['zz','zh','hh']:
            ax = hist.plot1d(hists['hists']['passPreSel']['fourTag']['SR']['trigWeight'][f'SvB_ps_{bb}'], overlay='trigWeight')
            fig = ax.get_figure()
            fig.savefig(f'SvB_ps_{bb}.pdf')
            fig.clear()

        ax = hist.plot1d(hists['hists']['passPreSel']['fourTag']['SR']['canJet.pt'], overlay='dataset')
        fig = ax.get_figure()
        fig.savefig('canJet_pt.pdf')
        fig.clear()

        ax = hist.plot1d(hists['hists']['passPreSel']['fourTag']['SR']['quadjet_selected.lead.mass'], overlay='dataset')
        fig = ax.get_figure()
        fig.savefig('quadjet_selected_lead_mass.pdf')
        fig.clear()

        ax = hist.plot1d(hists['hists']['passPreSel']['fourTag']['SR']['quadjet_selected.lead.dr'], overlay='dataset')
        fig = ax.get_figure()
        fig.savefig('quadjet_selected_lead_dr.pdf')
        fig.clear()


        ax = hist.plot1d(hists['hists']['passPreSel']['fourTag']['SR']['v4j.mass'], overlay='dataset')
        fig = ax.get_figure()
        fig.savefig('v4j_mass.pdf')
        fig.clear()




