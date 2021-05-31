import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


ON_EVENT_COLOR = np.array((191, 55, 80)) /255
OFF_EVENT_COLOR = np.array((75, 112, 173)) /255

def get_output_data(name, unserialize_yx=True):
    data = pickle.load(open(f'../data/events_{name}.pkl', 'rb'))
    if unserialize_yx:
        for i, (mon, events) in enumerate(data.items()):
            nids, ts = events.T
            yx_nids = np.array((nids//8, nids%8))
            if mon != 'DVS':
                ts -= ts[0]
            data[mon] = np.concatenate((yx_nids, ts[np.newaxis,:])).T
    return data

def plot_firing_rates(data):
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
    plt.savefig('../data/heatmaps.png')

def plot_dynamic_response(data, T=16000, fps=None):
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    [spine.set_visible(False) for ax in axes.flatten() for spine in ax.spines.values()]
    fig.subplots_adjust(top=.95, bottom=.05, wspace=.2, hspace=.18, left=.02, right=.7)
    mpl_frames = []

    last_t = max([dat[-1,2] for dat in data.values()])
    nframes = last_t // T
    if fps is None:
        # set fps to match actual raw recording length
        fps = int(1*1e6/T)
        
    # the video is made with matplotlibs animations API
    # this collects a list for each frame containing the heatmap and a text label artist
    bin_range = (0,0)
    for i in range(nframes):
        # move the bin boundary by the periodlength
        bin_range = bin_range[1], bin_range[1]+T
        
        lbl = f't: {bin_range[1]/1e6:.2f} s,   T: {T//1e3} ms,   FPS: {fps}'
        text = axes[0,0].text(0.73,.9, lbl, transform=fig.transFigure, fontsize=14)
        one_frame_artists = [text]

        for i, (mon, events) in enumerate(data.items()):
            # make an empty whitish frame
            frame = np.full((8,8,3), 240, np.ubyte)
            y, x, t = events.T

            if mon == 'DVS':
                row, column = 0, 0
                title = 'DVS camera input'
                arr_stop, arr_start, col = (1.03,0.1), (1.17, 0.1), ON_EVENT_COLOR
            elif mon == 'WTA':
                row, column = 0, 1
                title = 'WTA population'
                arr_stop, arr_start, col = (-.05, -.03), (-.2,-.18), ON_EVENT_COLOR
            elif mon == 'INH':
                row, column = 1, 0
                title = 'Inhibitory population'
                arr_stop, arr_start, col = (1.03, 1.03), (1.18,1.18), OFF_EVENT_COLOR
            elif mon == 'STIM':
                title = 'stimulating (output) population'
                row, column = 1, 1
                arr_stop, arr_start, col = (0.1, 1.17), (0.1,1.03), ON_EVENT_COLOR
            else:
                continue
            ax = axes[row, column]
            ax.annotate('', xy=arr_stop, xycoords='axes fraction', xytext=arr_start, 
                        arrowprops=dict(arrowstyle="<|-", linewidth=2, color=col))
            ax.set_title(title)
            ax.axis('off')

            # slice to timepoints that are in the timebin
            idx_in_bin = np.argwhere((bin_range[0] <= t) & (t < bin_range[1]))[:,0]
            
            # draw events if any for that timerange/ frame
            if idx_in_bin.any():
                # slice coordinates to timebins, set their color in the video
                coo = np.stack((y[idx_in_bin], x[idx_in_bin]))
                frame[coo[0], coo[1], :] = col

            # draw heatmap
            drawn_frame = ax.imshow(frame, animated=True)
            one_frame_artists.append(drawn_frame)

            # # spines
        mpl_frames.append(one_frame_artists)

    ani = animation.ArtistAnimation(fig, mpl_frames, interval=T//1000, blit=True, repeat_delay=1000)
    plt.show()
    # exit()
    print('encoding video...', end='')
    Writer = animation.writers['ffmpeg'](fps=fps)
    ani.save('../data/output.mp4', writer=Writer)
    # print(f'Saved: {}')


def main():
    name = 'DAVIS1'
    # data = get_output_data(name, unserialize_yx=False)
    # plot_firing_rates(data)
    
    data = get_output_data(name, unserialize_yx=True)
    plot_dynamic_response(data, T=41000)


if __name__ == '__main__':
    main()