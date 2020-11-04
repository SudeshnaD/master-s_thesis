# Merge all text files generated

import glob

with open('All_avg_meanavg','w') as outfile:
    for name in glob.glob('avg_meanavg*'):
        with open(name,'r') as infile:
            outfile.write(infile.read())

with open('All_FTavg_meanavg','w') as outfile:
    for name in glob.glob('FTavg_meanavg*'):
        with open(name,'r') as infile:
            outfile.write(infile.read())