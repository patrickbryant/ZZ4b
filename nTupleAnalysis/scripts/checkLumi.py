from glob import glob

logs = glob('condor_log_data201*.stdout')
logs.sort()

# lumiDict = {
#     # Old lumi
#     '2016':  '36.3e3',
#     '2016_preVFP': '19.5e3',
#     '2016_postVFP': '16.5e3',
#     '2017':  '36.7e3',
#     '2018':  '59.8e3',
#     'RunII':'132.8e3',
#     # Updated lumi with name change trigger from 2017 and btag change trigger from 2018
#     # '2016':  '36.5e3',
#     # '2017':  '41.5e3',
#     # '2018':  '60.0e3',
#     # '17+18':'101.5e3',
#     # 'RunII':'138.0e3',
# }

# expected_lumi   = {'2016':   36.5,#35.8791
#                    '2017':   41.5,#36.7338
#                    '2018':   60.0,#59.9656
#                    'RunII': 138.0}

expected_lumi   = {'2016':   36.3,#35.8791
                   '2017':   36.7,#36.7338
                   '2018':   59.8,#59.9656
                   'RunII': 132.8}

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
