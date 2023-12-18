import numpy as np
import pandas as pd

arr = np.array([[100, 200, 200, 50], [90, 200, 300, 0], [5, 200, 10, 200]])
data = pd.DataFrame(arr, index=['DevOpsHandbook', 'ContinuousDelivery', 'DistributedComputing'],
                    columns=['business', 'computer', 'git', 'parallel'])
data = np.log10(data + 1)

print(data.loc['DevOpsHandbook'])
print(data.loc['ContinuousDelivery'])
print(data.loc['DistributedComputing'])

print(data.loc['DevOpsHandbook'].dot(data.loc['ContinuousDelivery']))
print(data.loc['DevOpsHandbook'].dot(data.loc['DistributedComputing']))
print(data.loc['ContinuousDelivery'].dot(data.loc['DistributedComputing']))

data.loc['DevOpsHandbook'] /= np.sqrt((data.loc['DevOpsHandbook'] ** 2).sum())
data.loc['ContinuousDelivery'] /= np.sqrt((data.loc['ContinuousDelivery'] ** 2).sum())
data.loc['DistributedComputing'] /= np.sqrt((data.loc['DistributedComputing'] ** 2).sum())
print(data.to_markdown())
