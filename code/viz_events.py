import pickle
import numpy as np
import matplotlib.pyplot as plt

name = 'DAVIS1'
data = pickle.load(open(f'../data/events_{name}.pkl', 'rb'))
if 'WTA' not in data:
    data['WTA'] = np.empty((0,2))
frates = np.zeros((len(data), 8, 8))

fig, axes = plt.subplots(2,3,figsize=(15,10))
axes[1,2].axis('off')
for i, (mon, events) in enumerate(data.items()):
    nids, ts = events.T
    non_zero_nids = np.unique(nids)

    for non_zero_nid in non_zero_nids:
        y, x = non_zero_nid//8, non_zero_nid%8
        frates[i, y, x] = np.sum(nids==non_zero_nid)

    if mon == 'DVS':
        row, column = 1, 0
        arr_stop, arr_start, col = (.5,1.1), (.5, 1.1), 'r'
    if mon == 'FPGA':
        row, column = 0, 0
        arr_stop, arr_start, col = (1.3, .5), (1.42,.5), 'r'
    if mon == 'WTA':
        row, column = 0, 1
        arr_stop, arr_start, col = (.5, -.1), (.5,-.3), 'r'
    if mon == 'INH':
        row, column = 1, 1
        arr_stop, arr_start, col = (.4, 1.2),(.4,1.4), 'b'
    if mon == 'STIM':
        row, column = 0, 2
        arr_stop, arr_start, col = (-.1, .5), (-.3,.5), 'r'
    kwargs = {'vmin':0, 'vmax':1} if not frates[i].any() else {}
    ax = axes[row, column]
        
    ax.set_title(f'Spikecounts {mon}', fontsize=10)
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.tick_params(labelleft=False, labelbottom=False, top=True, right=True)
    ax.vlines(np.arange(.5, 8.5), np.full(8, -.5), np.full(8, 8.5), color='k', linewidth=.4, zorder=5)
    ax.hlines(np.arange(.5, 8.5), np.full(8, -.5), np.full(8, 8.5), color='k', linewidth=.4, zorder=5)

    im = ax.imshow(frates[i], cmap='OrRd', **kwargs)
    fig.colorbar(im, ax=ax, shrink=.8)
    
    # ax.arrow(1, 1, 2, 2, head_width=1, head_length=2, fc='k', ec='k', clip_on=False)
    ax.annotate('', xy=arr_stop, xycoords='axes fraction', xytext=arr_start, 
                arrowprops=dict(arrowstyle="<|-", color=col, linewidth=3))
plt.show()