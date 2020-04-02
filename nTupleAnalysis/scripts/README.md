Instructions on making 4b presentation

Initial Setup (Clone ROOTHelp)

> git clone git@github.com:johnalison/ROOTHelp.git

Only do this once. 

Then everytime you sign on and want to make plots do

> sourse ROOTHelp/setup.sh

First make the all the pdfs

> python make4bPlots.py  path/to/histA.root path/to/histB.root    --out plots_AvsB

This produced a directory named plots_AvsB with a bunch of plots 

To collect all the plots into one PDF do

> py make4bPresentation.py  -n comp_AvsB  -d plots_AvsB


Can also give 
 -t three
to do the comparison in the three tag region