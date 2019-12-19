import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inputFile',           help="Run in loop mode")
o, a = parser.parse_args()


totalHemis_3Tag = 0
maxNHemis_3Tag = 0
totalHemis_4Tag = 0
maxNHemis_4Tag = 0

isThreeTag = False
isFourTag  = False

inFile = open(o.inputFile,"r")
for line in inFile.readlines():
    words = line.split()

    if 'Processing file:' in line:
        if '3TagEvents' in line:
            isThreeTag = True
            isFourTag = False
        if '4TagEvents' in line:
            isThreeTag = False
            isFourTag = True
            
    assert(not (isThreeTag and isFourTag))

    if "nHemis" in words:
        thisNHemi = int(words[0])
        if isThreeTag:
            totalHemis_3Tag += thisNHemi
            if thisNHemi > maxNHemis_3Tag:
                maxNHemis_3Tag = thisNHemi
        elif isFourTag:
            totalHemis_4Tag += thisNHemi
            if thisNHemi > maxNHemis_4Tag:
                maxNHemis_4Tag = thisNHemi            


print "Total number of hemispheres loaded",totalHemis_3Tag + totalHemis_4Tag        
print "Max number of hemispheres loaded from one file",max(maxNHemis_3Tag,maxNHemis_4Tag)
print "\tTotal number of 4-tag hemispheres loaded",totalHemis_4Tag
print "\tMax number of 4-tag hemispheres loaded from one file",maxNHemis_4Tag
print "\tTotal number of 3-tag hemispheres loaded",totalHemis_3Tag
print "\tMax number of 3-tag hemispheres loaded from one file",maxNHemis_3Tag
        
