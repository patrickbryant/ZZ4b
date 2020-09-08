import ROOT

base="/uscms/home/bryantp/nobackup/ZZ4b/"
data=base+"dataRunII/hists_j_r.root"
ttbar=base+"TTRunII/hists_j_r.root"
zh=base+"bothZH4bRunII/hists.root"
zz=base+"ZZ4bRunII/hists.root"

data=ROOT.TFile(data,"READ")
ttbar=ROOT.TFile(ttbar,"READ")
zh=ROOT.TFile(zh,"READ")
zz=ROOT.TFile(zz,"READ")

data3b_zh=data.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zh")
ttbar4b_zh=ttbar.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zh")
ttbar3b_zh=ttbar.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zh")
zh4b_zh=zh.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zh")
zz4b_zh=zz.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zh")
zh3b_zh=zh.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zh")
zz3b_zh=zz.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zh")

a=data3b_zh.FindBin(0.95)
b=data3b_zh.FindBin(1)
ndata3b_zh=data3b_zh.Integral(a,b)
nttbar4b_zh=ttbar4b_zh.Integral(a,b)
nttbar3b_zh=ttbar3b_zh.Integral(a,b)
nb4b_zh=ndata3b_zh+nttbar4b_zh
nzh4b_zh=zh4b_zh.Integral(a,b)
nzz4b_zh=zz4b_zh.Integral(a,b)
ns4b_zh=nzh4b_zh+nzz4b_zh
nzh3b_zh=zh3b_zh.Integral(a,b)
nzz3b_zh=zz3b_zh.Integral(a,b)
ns3b_zh=nzh3b_zh+nzz3b_zh
r4b_zh=ns4b_zh/nb4b_zh
r3b_zh=ns3b_zh/ndata3b_zh

data3b_zz=data.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zz")
ttbar4b_zz=ttbar.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zz")
ttbar3b_zz=ttbar.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zz")
zh4b_zz=zh.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zz")
zz4b_zz=zz.Get("passMDRs/fourTag/mainView/SR/SvB_ps_zz")
zh3b_zz=zh.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zz")
zz3b_zz=zz.Get("passMDRs/threeTag/mainView/SR/SvB_ps_zz")

a=data3b_zz.FindBin(0.95)
b=data3b_zz.FindBin(1)
ndata3b_zz=data3b_zz.Integral(a,b)
nttbar4b_zz=ttbar4b_zz.Integral(a,b)
nttbar3b_zz=ttbar3b_zz.Integral(a,b)
nb4b_zz=ndata3b_zz+nttbar4b_zz
nzh4b_zz=zh4b_zz.Integral(a,b)
nzz4b_zz=zz4b_zz.Integral(a,b)
ns4b_zz=nzh4b_zz+nzz4b_zz
nzh3b_zz=zh3b_zz.Integral(a,b)
nzz3b_zz=zz3b_zz.Integral(a,b)
ns3b_zz=nzh3b_zz+nzz3b_zz
r4b_zz=ns4b_zz/nb4b_zz
r3b_zz=ns3b_zz/ndata3b_zz

print "3b zh | zh   | zz   | sig. | mult.   |  ttbar | bkgd    | signal fraction"
print "      | %4.1f | %4.1f | %4.1f |     N/A | %6.1f | %7.1f | %f "%(nzh3b_zh, nzz3b_zh, ns3b_zh, nttbar3b_zh, ndata3b_zh, r3b_zh)
print
print "4b zh | zh   | zz   | sig. | mult.   |  ttbar | bkgd    | signal fraction"
print "      | %4.1f | %4.1f | %4.1f | %7.1f | %6.1f | %7.1f | %f "%(nzh4b_zh, nzz4b_zh, ns4b_zh, ndata3b_zh, nttbar4b_zh, nb4b_zh, r4b_zh)
print
print "     f3b/f4b | f4b/f3b"
print "     %6.1f%% | %6.1fx"%(100*r3b_zh/r4b_zh, r4b_zh/r3b_zh)
print
print "     S3b/S4b | S4b/S3b"
print "     %6.1f%% | %6.1fx"%(100*ns3b_zh/ns4b_zh, ns4b_zh/ns3b_zh)
print "-"*50
print "3b zz | zh   | zz   | sig. | mult.   |  ttbar | bkgd    | signal fraction"
print "      | %4.1f | %4.1f | %4.1f |     N/A | %6.1f | %7.1f | %f "%(nzh3b_zz, nzz3b_zz, ns3b_zz, nttbar3b_zz, ndata3b_zz, r3b_zz)
print
print "4b zz | zh   | zz   | sig. | mult.   |  ttbar | bkgd    | signal fraction"
print "      | %4.1f | %4.1f | %4.1f | %7.1f | %6.1f | %7.1f | %f "%(nzh4b_zz, nzz4b_zz, ns4b_zz, ndata3b_zz, nttbar4b_zz, nb4b_zz, r4b_zz)
print
print "     f3b/f4b | f4b/f3b"
print "     %6.1f%% | %6.1fx"%(100*r3b_zz/r4b_zz, r4b_zz/r3b_zz)
print
print "     S3b/S4b | S4b/S3b"
print "     %6.1f%% | %6.1fx"%(100*ns3b_zz/ns4b_zz, ns4b_zz/ns3b_zz)
