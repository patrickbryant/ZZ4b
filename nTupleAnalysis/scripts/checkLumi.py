from glob import glob

logs = glob('condor_log_data201*.stdout')
logs.sort()

expected_lumi   = {'2016':   36.5,#35.8791
                   '2017':   41.5,#36.7338
                   '2018':   60.0,#59.9656
                   'RunII': 138.0}

year_lumi = {'2016': 0.0,
             '2017': 0.0,
             '2018': 0.0}

total_lumi = 0.0
for logFile in logs:
    period = logFile.split('_')[2].replace('data','')
    if len(period)<5: continue
    year = period[0:4]

    with open(logFile) as log:
        # lines = log.readlines()
        last_line = log.readlines()[-2].replace('\n','')
        try:
            index = last_line.find('/fb')
            lumi = float(last_line[:index].split()[-1].replace('/fb)',''))
        except:
            lumi = 0.0
            print('ERROR: %s'%(logFile))
            print('       %s'%(last_line))
        year_lumi[year] += lumi
        total_lumi += lumi
        print(' %s %6.2f/fb'%(period, lumi))
    
print('----------------')
for year in ['2016', '2017', '2018']:
    print(' %s: %6.2f/fb (%6.2f/fb expected)'%(year, year_lumi[year], expected_lumi[year]))
print('Total: %6.2f/fb (%6.2f/fb expected)'%(total_lumi, expected_lumi['RunII']))
