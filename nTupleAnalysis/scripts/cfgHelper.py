import os

def mkpath(path):
    print "mkpath",path
    dirs = path.split("/")
    thisDir = ""
    for d in dirs:
        thisDir = thisDir+d+"/"
        if not os.path.exists(thisDir):
            os.mkdir(thisDir)
