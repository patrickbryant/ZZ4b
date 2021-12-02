import os
import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inputFiles',     help="Input files")
parser.add_option('-n', '--names',          help="Legend Names")
parser.add_option('-o', '--outdir',     default='', type=str, help='outputDirectory')
o, a = parser.parse_args()

outputDir = o.outdir
if not os.path.isdir(outputDir):
    print("Making output dir",outputDir)
    os.mkdir(outputDir)


print(o.inputFiles)

#"ZZ4b/nTupleAnalysis/pytorchModels/3bMix4bv1FvT_ResNet+multijetAttention_9_9_9_np2070_lr0.008_epochs40_stdscale.log"]
inputFileNames = o.inputFiles.split(",")
labels         = o.names.split(",")

if not len(inputFileNames) == len(labels):
    print("Number of input files and name have to be the same!")
    print("you gave ",len(inputFileNames),"vs", len(labels))
    print("Exiting...")
#for i in o.inputFiles.split.:
#    inputFiles.append(i)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def readLogFile(infileName,label, debug = False):
    print("Processing",infileName)
    infile = open(infileName,"r")
    
    arrowNum = 1
    epochNum = 2
    trainNum = 4
    valNum   = 1
    controlNum = 1

    for line in infile:
        words = line.split()

        if len(words) < 2: continue


        if words[0][0] == "#": continue

        if words[arrowNum] == ">>" and words[epochNum] == "Epoch":
            data={}
            data["file"] = infileName.split("/")[-1]
            data["label"] = label
            data["epochs"] = []

            data["val_loss"] = []
            data["val_norm"] = []
            data["val_AUC_d43"] = []
            data["val_AUC_td"] = []

            data["ABC"] = []
            data["chi2perBin"] = []

            data["train_loss"] = []
            data["train_norm"] = []
            data["train_AUC_d43"] = []
            data["train_AUC_td"] = []

            data["control_loss"] = []
            data["control_norm"] = []
            data["control_AUC_d43"] = []
            data["control_AUC_td"] = []


            epoch = 0
            continue

        try: 

            if words[arrowNum] == ">>" and words[trainNum] == "Training":        
                epoch = int(words[epochNum].split("/")[0])
                data["epochs"].append(epoch)                        
                if debug: print("Reading epoch", epoch,"Training" )
    
                data["train_loss"].append(float(words[6]))
                data["train_norm"].append(float(words[12]))
                data["train_AUC_d43"].append(float(words[16]))
                data["train_AUC_td"].append(float(words[18]))
    
    
    
            if words[valNum] == "Validation":                    
                if debug: print("Reading epoch", epoch,"Validation" )
    
                data["val_loss"].append(float(words[3]))
                data["val_norm"].append(float(words[9]))
                data["val_AUC_d43"].append(float(words[13]))
                data["val_AUC_td"].append(float(words[15]))
    
                #if epoch == 0:
                #    data["overTrain"].append(0)
                #    data["chi2perBin"].append(0)
                #else:
                data["ABC"].append(float(words[18].replace("%","").replace("(","").replace(",","") ))
                data["chi2perBin"].append(float(words[20].replace(",","")))
    
            if words[controlNum] == "Control":                    
                if debug: print("Reading epoch", epoch,"Control" )
                data["control_loss"].append(float(words[3]))
                data["control_norm"].append(float(words[9]))
                data["control_AUC_d43"].append(float(words[13]))
                data["control_AUC_td"].append(float(words[15]))
    
        except:
            print("ERROR on ")
            print(words)

    return data
            

def makePlot(name,inputData,xKey,yKey,estart,yTitle,logy=False,xTitle="Epoch",yMax=None,yMin=None):
    plt.figure(figsize=(10,7))
    if logy:
        plt.yscale('log')


    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    if yMax:
        plt.ylim(yMin,yMax)


    for ds in range(len(inputData)):

        lineStyle = ":"
        if ds % 2 == 0: lineStyle = "-."
        if not ds: lineStyle="-"


        if not ds:
            plt.plot(inputData[0][xKey][estart:], inputData[0][yKey][estart:],marker="o",linestyle=lineStyle,color="r",label=inputData[0]["label"])
            #plt.plot(inputData[0][xKey][estart:], inputData[0][yKey][estart:],marker="o",linestyle=":",color="r",label=inputData[0]["label"])
        else:
            #if ds == 7:
             #   plt.plot(inputData[ds][xKey][estart:], inputData[ds][yKey][estart:],marker="o",linestyle="-",color="r",label=inputData[ds]["label"])
            #else:
            plt.plot(inputData[ds][xKey][estart:], inputData[ds][yKey][estart:],marker="o",linestyle=lineStyle,label=inputData[ds]["label"])
    plt.legend(loc="best")
    plt.savefig(outputDir+"/"+name+".pdf")


def plotData(inputData,estart=0):

    makePlot("Val_Loss",  inputData,"epochs","val_loss",  estart,"Validation Loss", yMin=0.85,yMax=0.93)
    makePlot("Val_Loss_l",  inputData,"epochs","val_loss",  estart,"Validation Loss")
    makePlot("Val_Norm",  inputData,"epochs","val_norm",  estart,"Validation Norm",yMin=0.5, yMax=2)
    makePlot("Val_AUC_d43",   inputData,"epochs","val_AUC_d43",   estart,"Validation AUC", yMin=70,yMax=90)
    makePlot("Val_AUC_td",   inputData,"epochs","val_AUC_td",   estart,"Validation AUC", yMin=62,yMax=70)

    makePlot("Train_Loss",  inputData,"epochs","train_loss",estart,"Training Loss", yMin=0.85,yMax=0.93)
    makePlot("Train_Loss_l",inputData,"epochs","train_loss",estart,"Training Loss")#,logy=True)
    makePlot("Train_Norm",  inputData,"epochs","train_norm",estart,"Training Norm",yMin=0.5, yMax=2)
    makePlot("Train_AUC_d43",   inputData,"epochs","train_AUC_d43", estart,"Training AUC", yMin=70,yMax=90)
    makePlot("Train_AUC_td",   inputData,"epochs","train_AUC_td", estart,"Training AUC", yMin=62,yMax=70)

    #makePlot("control_Loss",   inputData,"epochs","control_loss",estart,"Control Loss", yMin=0.85,yMax=0.93)
    #makePlot("control_Loss_l", inputData,"epochs","control_loss",estart,"Control Loss")#,logy=True)
    #makePlot("control_Norm",   inputData,"epochs","control_norm",estart,"Control Norm",yMin=0.5, yMax=2)
    #makePlot("control_AUC_d43",inputData,"epochs","control_AUC_d43", estart,"Control AUC", yMin=70,yMax=90)
    #makePlot("control_AUC_td", inputData,"epochs","control_AUC_td", estart,"Control AUC", yMin=62,yMax=70)


    makePlot("ABC", inputData,"epochs","ABC", estart,"ABC", yMin=0, yMax=10)
    makePlot("chi2perBin", inputData,"epochs","chi2perBin", estart,"Chi2 per Bin", yMin=0.5, yMax=3.5)



inputData = []

for itr, inName in enumerate(inputFileNames):
    inputData.append(readLogFile(inName,labels[itr]))





plotData(inputData,estart=1)

#print(inputData)
#plt.ylim(ratioRange)
#plt.xlim([bins[0],bins[-1]])
#plt.plot([bins[0], bins[-1]], [1, 1], color='k', alpha=0.5, linestyle='--', linewidth=1)

