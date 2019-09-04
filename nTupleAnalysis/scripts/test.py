import matplotlibHelpers as pltHelper
import numpy as np
args={'points': np.random.rand(7000),
      'weights': np.random.rand(7000),
      'color': 'blue',
      'alpha': 1.0,
      'linewidth': 1,
      'name': 'Output',
      }
train=pltHelper.dataSet(**args)

args={'points': np.random.rand(3000),
      'weights': np.random.rand(3000),
      'color': 'blue',
      'alpha': 0.5,
      'linewidth': 2,
      }
valid=pltHelper.dataSet(**args)



args={'points': np.zeros(0),
      'weights': np.zeros(0),
      'color': 'black',
      'alpha': 1.0,
      'linewidth': 1,
      'name': 'Training Set',
      }
trainLegend=pltHelper.dataSet(**args)
args={'points': np.zeros(0),
      'weights': np.zeros(0),
      'color': 'black',
      'alpha': 0.5,
      'linewidth': 2,
      'name': 'Validation Set',
      }
validLegend=pltHelper.dataSet(**args)

args={'dataSets':[trainLegend,validLegend,train,valid], 
      'bins':[b/20.0 for b in range(21)], 
      'xlabel':'Random Number', 
      'ylabel':'Events',
      }
hist = pltHelper.histPlotter(**args)
print(hist.artists)
hist.artists[0].remove()
hist.artists[1].remove()
hist.savefig("../../../test.pdf")
