import pickle
import numpy as np
import matplotlib.pyplot as plt

name = 'DAVIS1'
data = pickle.load(open(f'../data/events_{name}.pkl', 'rb'))
print(data.keys())
print(len(data))
frates = np.zeros((len(data), 8, 8))

fig, axes = plt.subplots(1,3,figsize=(14,9))
for i, (mon, events) in enumerate(data.items()):
    nids, ts = events.T
    non_zero_nids = np.unique(nids)

    for non_zero_nid in non_zero_nids:
        y, x = non_zero_nid//8, non_zero_nid%8
        frates[i, y, x] = np.sum(nids==non_zero_nid)

    axes[i].imshow(frates[i], vmin=0, vmax=10)
    axes[i].set_title(f'Spikecounts {mon}')

plt.show()