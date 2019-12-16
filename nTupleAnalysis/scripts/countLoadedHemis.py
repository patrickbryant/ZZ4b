import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inputFile',           help="Run in loop mode")
o, a = parser.parse_args()


totalHemis = 0
maxNHemis = 0

inFile = open(o.inputFile,"r")
for line in inFile.readlines():
    words = line.split()
    if "nHemis" in words:
        thisNHemi = int(words[0])
        totalHemis += thisNHemi
        if thisNHemi > maxNHemis:
            maxNHemis = thisNHemi
print "Total number of hemispheres loaded",totalHemis        
print "Max number of hemispheres loaded from one file",maxNHemis
        
