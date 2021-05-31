import sys
import os
import glob
import pickle

import skvideo.io

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

from viz_dyn_output import ON_EVENT_COLOR, OFF_EVENT_COLOR

def from_numpy(datafile, norm_timestamps=True):
    if not os.path.exists(datafile):
        print('File not found: ', datafile)
        exit(1)
    t, x, y, pol = np.load(datafile).T
    if norm_timestamps:
        t -= t[0]
    return t, x, y, pol

def find_videofile(video_file):
    if not os.path.exists(video_file):
        video_file = video_file.replace('.mov', '.wmv')
        if not os.path.exists(video_file):
            video_file = video_file.replace('.wmv', '.mp4')
            if not os.path.exists(video_file):
                print('Video file does not exist.')
                exit(1)
    return video_file

def video2numpy(name, cut_to=None):
    print('Reading video as numpy array.')
    video_file = find_videofile(f'/home/loaloa/Videos/{name}.mov')
    
    video = skvideo.io.vread(video_file)
    nframes, H, W, _ = video.shape
    fps = int(skvideo.io.ffprobe(video_file)['video']['@avg_frame_rate'][:-2])
    T = 1/fps

    cut_mask = np.ones(nframes, bool)
    if cut_to is not None:
        frame_ts = np.arange(0,nframes*T, T)
        cut_mask = (frame_ts>cut_to[0]) & (frame_ts<cut_to[1])
        frame_ts = frame_ts[cut_mask]
    # framelength in u seconds
    T *= 1e6
    return video[cut_mask], fps, T, (H, W)

def reshape_events(t, x, y, pol, img_dim, downsample=False, crop_to=None, only_pol=None, get_kernel_viz_params=False):
    print('Reshaping event video (includes downsampling).')
    # limit the specific polarity, 0 or 1
    if only_pol is not None:
        idx_pol = np.where(pol==only_pol)[0]
        # slice to those
        t, y, x, pol = t[idx_pol], y[idx_pol], x[idx_pol], pol[idx_pol]
    
    # slice to some field of view
    if crop_to is not None:
        ymin,xmin, ymax,xmax = crop_to  
        img_dim = ymax-ymin, xmax-xmin
        # keep only indices that are in the cropped area
        idx_cropped = np.where((xmin<=x)  & (x<xmax) & (ymin<=y) & (y<ymax))[0]
        # slice to those
        t, y, x, pol = t[idx_cropped], y[idx_cropped], x[idx_cropped], pol[idx_cropped]
        # x and y coordinates need to be adjusted such that 0,0 is in the upper left corner
        y -= ymin
        x -= xmin

    # downsample the field of view to 8x8 pixels using simple concatenation
    if downsample:
        # get the input img dim
        ysize, xsize = img_dim
        # for final returned shape
        img_dim = 8,8
        # define the kernelsize such that we get exactly 8x8 subregions
        ykernelsize, xkernelsize = np.ceil(ysize/8), np.ceil(xsize/8)
        # to make the kernels fit, add 0 padding
        ypad, xpad = 8*ykernelsize -ysize, 8*xkernelsize -xsize
        ypad, xpad = (np.floor(ypad/2), np.ceil(ypad/2)), \
                     (np.floor(xpad/2), np.ceil(xpad/2))
        ypad, xpad = np.array(ypad, dtype=int), np.array(xpad, dtype=int)
        # special option only for visualizing the downsampling
        # no reshaping will be down with this option, only return kernel 
        if get_kernel_viz_params:
            return ypad, xpad, ykernelsize, xkernelsize

        # with the padding, the x and y coordinates need to be adjusted
        y += ypad[0]
        x += xpad[0]
        
        # ysize, xsize, ycol:start+stop , xcol:start+stop
        # kernel_coos is actually not used
        kernel_coos = np.zeros((8, 8, 2, 2), dtype=int)
        # kernel shape: y_start_coo, x_start_coo
        #               y_stop_coo,  x_stop_coo (excl)
        new_kernel = last_kernel = np.zeros((2,2), int)
        # build an 8x8 array with elements of shape 2,2. 2x2 is the kernel (see above)
        for row in range(8):
            print(f'{row*8}/64...', end='')
            if row:
                # index y start and y stop
                new_kernel[:,0] = last_kernel[:,0] + ykernelsize
            else:
                # index only y stop for first row
                new_kernel[1,0] = last_kernel[1,0] + ykernelsize

            for col in range(8):
                if col:
                    # index x start and x stop
                    new_kernel[:,1] = last_kernel[:,1] + xkernelsize
                else:
                    # reset x coordinates
                    new_kernel[:,1] = 0
                    # index only x stop for first column
                    new_kernel[1,1] = last_kernel[1,1] + xkernelsize
                # save the kernel, set it as the last kernel for next iteration
                kernel_coos[row,col] = new_kernel
                last_kernel = new_kernel

                # get the kernel coordinates
                ykernel_idcs = np.arange(last_kernel[0,0],last_kernel[1,0])
                xkernel_idcs = np.arange(last_kernel[0,1],last_kernel[1,1])
                # select the indices of x,y coordinates that are in the kernel area 
                in_kernel = [i for i, (y_val,x_val) in enumerate(zip(y,x)) \
                             if y_val in ykernel_idcs and x_val in xkernel_idcs]
                # those datapoints are simply set to the kernel coordinate (0 to 8, 0 to 8)
                x[in_kernel] = col
                y[in_kernel] = row
    print('Done.')
    return t, x, y, pol, img_dim

def serialize_nids(x, y, img_dim=(8,8)):
    return (img_dim[0]*y + x).astype(int)

def filter_events(t, x, y, pol, name, tau, thr):
    print('Filtering downsampled event video.')
    def exp_filter(t, pol, filter_floor=None, plot=False):
        # timestamps distance to current event
        t = (t-t[-1]) *-1
        # only pass sufficiently large values into the filter
        if filter_floor:
            suff_large = t<filter_floor
            t = t[suff_large]
            pol = pol[suff_large] 
        t = t /1e5 # deci-seconds
        # pass through filter and assign polarity 
        t_filt = np.exp(-t/tau) *pol
        
        if plot == 42:
            plot_exp_filter(t, t_filt, name, tau, thr)

        # only for first run where filter_floor is determined
        if filter_floor is None:
            return t_filt
        if np.abs(t_filt.sum()) > thr:
            return True
    
    # this idetifies the floor of the filter, the delta t where spikes are ignored, (speedup)
    test_timepoints = np.arange(1e4, 1e6, 1e4)
    filter_out = exp_filter(test_timepoints, pol=np.ones(len(test_timepoints)))
    filter_floor = test_timepoints[filter_out > .01][0]

    filtered_events = []
    for row in range(8):
        print(f'{(row+1)*8}/64...', end='')
        for col in range(8):
            pixel_mask = (y==row) & (x==col)
            ts = t[pixel_mask]
            pols = pol[pixel_mask]
            for i in range(len(ts)):
                if i and exp_filter(ts[:i].copy(), pol[:i].copy(), filter_floor, plot=i):
                    filtered_events.append([ts[i], col, row, pols[i]])
    print('Done.')
    data = np.array(filtered_events)
    return data[data[:,0].argsort()].T

def get_kernelsize(crop_to=None, img_dim=None):
    if crop_to is None and img_dim is not None:
        crop_to = (0,0, *img_dim)
    if crop_to:
        ymin,xmin, ymax,xmax = crop_to  
        size = ymax-ymin //8 * xmax-xmin //8
    return size

def find_filter_params(t, x, y, pol, name, kernelsize=None):
    taus = [.1, .3, .6, 1, 1.3, 1.6, 2.4, 3]
    thrs = [2,5,8,10, 20, 40, 70, 100]

    for tau in taus:
        for thr in thrs:
            data = filter_events(t, x, y, pol, name, tau, thr)
            tf, xf, yf, polf = data
            
            tag = f'_tau{tau}_thr{thr}_regionsize{kernelsize}'
            filtered_event_plot(t, x, y, pol, tf, xf, yf, polf, name=name, tag=tag)
            events2vid_plot(tf, xf, yf, polf, (8,8), fps=30, name=name, tag=tag)
            plot_isi(tf, xf, yf, polf, (8,8), name=name, tag=tag)

def plot_exp_filter(t, t_filt, name, tau, thr):
    plt.figure(figsize=(5,9))
    # setup spines
    [plt.gca().spines[at].set_visible(False) for at in ['top', 'right', 'bottom']]
    plt.gca().spines['left'].set_position(('data',-0.2))
    plt.hlines(0,-2.5,10, linewidth=.7)

    # setup axis limits
    plt.xlim(-1.5,10)
    plt.ylim(-thr-thr*.1,thr+thr*.1)
    
    # draw the exponential filter function 
    x = np.linspace(0,10,200)
    plt.plot(x, np.exp(-x/tau), color='k', linewidth=.7)
    plt.plot(x, -np.exp(-x/tau), color='k', linewidth=.7)
    
    # draw the filter output of the events using lines
    colors = [ON_EVENT_COLOR if ts > 0 else OFF_EVENT_COLOR 
              for ts in t_filt]
    plt.vlines(t, 0, t_filt, colors=colors, linewidth=.5)
    
    # draw the final filter output (sum of evaluations)
    plt.vlines(-1, 0, t_filt.sum(), linewidth=8, color='k', alpha=.6)
    # draw the threshold lines
    plt.axhline(thr, linewidth=3, color=ON_EVENT_COLOR, linestyle='--')
    plt.axhline(-thr, linewidth=3, color=OFF_EVENT_COLOR, linestyle='--')
    
    # annotate
    plt.text(-3, 0, 'SUM', fontsize=13, clip_on=False)
    plt.xlabel('time post spike [100ms]')
    plt.title(f'tau: {tau}, threshold: {thr}')
    plt.savefig(f'../data/vid2e/{name}/ExpFilter_example.png')
    plt.show
    plt.close()
    
def filtered_event_plot(t, x, y, pol, t_filt=None, x_filt=None, y_filt=None, pol_filt=None, name=None, tag=''):
    fig, axes = plt.subplots(8,8, figsize=(18,9), sharey=True, sharex=True)
    fig.subplots_adjust(left=.06, right=.99, top=.98, bottom=.12, hspace=.4, wspace=.08)

    nid = serialize_nids(x, y)
    if x_filt is not None:
        nid_filt = serialize_nids(x_filt, y_filt)

    pixel_id = 0
    for row in range(8):
        for col in range(8):
            ax = axes[row, col]
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax.set_ylabel(pixel_id, labelpad=-10, rotation=0, fontsize=12, y=.95)
            [ax.spines[at].set_visible(False) for ax in axes.flatten() for at in ['top', 'right']]
            ax.set_ylim(0,2)
            
            if row == 7:
                ax.tick_params(bottom=True, labelbottom=True)
                ax.set_xlabel('[s]')
                if col == 3:
                    lbl = tag
                    if t_filt is not None:
                        reduc = (1 - t_filt.shape[0]/t.shape[0]) *100
                        lbl = f'{tag} reduction: {reduc:.2f}%'
                    ax.text(1.5, -1.8, lbl, clip_on=False,
                            fontsize=14)
            if col == 0:
                ax.annotate('filtered:\n\noriginal:', (-0.3, .25), xycoords='axes fraction', clip_on=False,
                        fontsize=10)
                
            data = t[nid==pixel_id] /1e6
            colors = [ON_EVENT_COLOR if p == 1 
                      else OFF_EVENT_COLOR 
                      for p in pol[nid==pixel_id]]
            ax.vlines(data, 0.2,0.8, linewidths=.5, colors=colors)
            
            if t_filt is not None:
                data = t_filt[nid_filt==pixel_id] /1e6
                colors = [ON_EVENT_COLOR if p == 1 
                          else OFF_EVENT_COLOR 
                          for p in pol_filt[nid_filt==pixel_id]]
                ax.vlines(data, 1.2,1.8, linewidths=.5, colors=colors)

            pixel_id += 1
    fname = f'../data/vid2e/{name}/filtered_events{tag}.png'
    # plt.show()
    fig.savefig(fname)
    plt.close()

def plot_isi(t, x, y, pol, img_dim, name, tag=''):
    pixelid = serialize_nids(x,y, img_dim=img_dim)
    data = np.stack([pixelid, t, pol], axis=1)
    data = data[data[:,0].argsort()]

    unique_pixels = np.unique(data[:,0])
    # on-on, off-off, on-off isis
    isis = [[],[],[]]
    n = 10000 if len(unique_pixels) > 10000 else len(unique_pixels)
    for pixel in np.random.choice(unique_pixels, size=n, replace=False):
        pixel_events = data[data[:,0]==pixel]
        pixel_events = pixel_events[np.argsort(pixel_events[:,1])]

        on_isis = (pixel_events[:-1,2]==1)  &  (pixel_events[1:,2]==1)
        off_isis = (pixel_events[:-1,2]==-1)  &  (pixel_events[1:,2]==-1)
        onoff_isis = (pixel_events[:-1,2] != pixel_events[1:,2])

        pixel_isis = pixel_events[1:,1] - pixel_events[:-1,1] 
        isis[0].extend(list(pixel_isis[on_isis]))
        isis[1].extend(list(pixel_isis[off_isis]))
        isis[2].extend(list(pixel_isis[onoff_isis]))
    
    plt.yscale('log')
    plt.hist(isis[0], bins=150, label='isi between on events', alpha=.8, color=ON_EVENT_COLOR, range=(0,1*1e6))
    plt.hist(isis[1], bins=150, label='isi between off events', alpha=.8, color=OFF_EVENT_COLOR, range=(0,1*1e6))
    plt.hist(isis[2], bins=50, label='isi between on-, and off events', alpha=.25, color='w', edgecolor='k', linewidth=.5, range=(0,1*1e6))
    plt.title(tag)
    plt.legend()
    plt.ylabel('count')
    plt.xlabel('inter spike interval in us')
    # plt.show()
    fname = f'../data/vid2e/{name}/isi{tag}.png'
    plt.savefig(fname, dpi=300)
    plt.close()

def events2vid_plot(t, x, y, pol, img_dim, name, T=None, fps=None, tag='',
                    viz_reshape=None):
    if T is None:
        # set frame duration to match fps
        T = int(1*1e6 / fps)
    if fps is None:
        # set fps to match actual raw recording length
        fps = int(1*1e6/T)
    nframes = int(t[-1] // T)
        
    # if kernel areas are drawn, unpack the viz_kernel list, pad video + x & y
    if viz_reshape is not None:
        ymin,xmin, ymax,xmax = viz_reshape

    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_title(name, pad=60, fontsize=15)
    ax.axis('off')
    if not 1 in pol:
        polarities = '[only off events]'
    elif not 0 in pol:
        polarities = '[only on events]'
    else:
        polarities = '[both on & off events]'

    # this collects a list for each frame containing the heatmap and a text label artist
    mpl_frames = []
    bin_range = (0,0)
    for fcount in range(nframes):
        frame = np.full((*img_dim, 3), 240, np.ubyte)
        if not fcount % 10:
            print(f'frame {fcount}/{nframes}...', end='')
        # move the bin boundary by the periodlength
        bin_range = bin_range[1], bin_range[1]+T
        # slice to timepoints that are in the timebin
        idx_in_bin = np.argwhere((bin_range[0] <= t) & (t < bin_range[1]))[:,0]
        
        # draw events if any for that timerange/ frame
        if idx_in_bin.any():
            frame_pol_color = [img_as_ubyte(ON_EVENT_COLOR) if p == 1 
                               else img_as_ubyte(OFF_EVENT_COLOR) 
                               for p in pol[idx_in_bin]]
            # slice coordinates to timebins, set their color in the video
            coo = np.stack((y[idx_in_bin], x[idx_in_bin]))
            frame[coo[0], coo[1], :] = frame_pol_color

        # draw kernel spines
        if viz_reshape is not None:
            y_coos = np.linspace(ymin,ymax,9)
            x_coos = np.linspace(xmin,xmax,9)
            ax.vlines(x_coos, np.full(9, ymin), np.full(9, ymax), color='grey', linewidth=.5),
            ax.hlines(y_coos, np.full(9, xmin), np.full(9, xmax), color='grey', linewidth=.5),

        # draw heatmap
        frame = ax.imshow(frame, animated=True)
        # labelling
        lbl = f't: {bin_range[1]/1e6:.2f} s, T: {T//1e3} ms, FPS: {fps}, polarities: {polarities}'
        text = ax.text(0,1.05, lbl, transform=ax.transAxes, fontsize=13)

        mpl_frames.append([frame, text])
    ani = animation.ArtistAnimation(fig, mpl_frames, interval=T//1000, blit=True)
    # plt.show()
    print('encoding video...', end='')
    Writer = animation.writers['ffmpeg'](fps=fps)
    fname = f'../data/vid2e/{name}/events{tag}.mp4'
    ani.save(fname, writer=Writer)
    print(f'Saved: {fname}')
    plt.close()



def plot_stimulus_preprocessing(data, fps, T, img_dim, name=None, viz_reshape=None, 
                                slowmo=False, tag=''):
    
    video, raw_events, downs_events, filtd_events = data 
    if slowmo:
        # upsampling like a pro
        T = T //2
        video = np.concatenate([np.stack([fr,fr]) for fr in video])
    nframes = video.shape[0]
    H, W = img_dim
    if viz_reshape is not None:
        ymin,xmin, ymax,xmax = viz_reshape
        
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    [spine.set_visible(False) for ax in axes.flatten() for spine in ax.spines.values()]
    fig.subplots_adjust(top=.95, bottom=.05, wspace=.2, hspace=.18, left=.02, right=.7)
    mpl_frames = []

    # the video is made with matplotlibs animations API
    # this collects a list for each frame containing the heatmap and a text label artist
    bin_range = (0,0)
    for fcount in range(nframes):
        if not fcount % 10:
            print(f'frame {fcount}/{nframes}...', end='')
        # move the bin boundary by the periodlength
        bin_range = bin_range[1], bin_range[1]+T
        
        lbl = f't: {bin_range[1]/1e6:.2f} s,   T: {T//1e3} ms,   FPS: {fps}'
        text = axes[0,0].text(0.73,.9, lbl, transform=fig.transFigure, fontsize=14)
        one_frame_artists = [text]

        for i, data in enumerate((video, raw_events, downs_events, filtd_events)):
            
            kwargs = {}
            if i == 0:
                row, column = 0, 0
                img_dim = H, W
                kwargs = {'cmap': 'gray'}
                title = 'Webcam video'
            elif i == 1:
                row, column = 0, 1
                img_dim = H, W
                title = 'Simulated events'
                if viz_reshape is not None:
                    y_coos = np.linspace(ymin,ymax,9)
                    x_coos = np.linspace(xmin,xmax,9)
                    axes[row,column].vlines(x_coos, np.full(9, ymin), np.full(9, ymax), color='grey', linewidth=.5),
                    axes[row,column].hlines(y_coos, np.full(9, xmin), np.full(9, xmax), color='grey', linewidth=.5),
            elif i == 2:
                row, column = 1, 0
                img_dim = 8, 8
                title = 'Downscaled'
            elif i == 3:
                row, column = 1, 1
                img_dim = 8, 8
                title = 'Filtered'
            else:
                continue

            if i:
                # make an empty whitish frame
                frame = np.full((*img_dim, 3), 240, np.ubyte)
                t, x, y, pol = data
                # slice to timepoints that are in the timebin
                idx_in_bin = np.argwhere((bin_range[0] <= t) & (t < bin_range[1]))[:,0]

                # draw events if any for that timerange/ frame
                if idx_in_bin.any():
                    frame_pol_color = [img_as_ubyte(ON_EVENT_COLOR) if p == 1 
                                       else img_as_ubyte(OFF_EVENT_COLOR) 
                                       for p in pol[idx_in_bin]]
                    # slice coordinates to timebins, set their color in the video
                    coo = np.stack((y[idx_in_bin], x[idx_in_bin]))
                    frame[coo[0], coo[1], :] = frame_pol_color
            else:
                # frame = rgb2gray(data[fcount])
                frame = data[fcount]
                
            ax = axes[row, column]
            ax.set_title(title)
            ax.axis('off')

            # draw heatmap
            drawn_frame = ax.imshow(frame, animated=True, **kwargs)
            one_frame_artists.append(drawn_frame)
        mpl_frames.append(one_frame_artists)

    ani = animation.ArtistAnimation(fig, mpl_frames, interval=T//1000, blit=True, repeat_delay=1000)
    # plt.show()
    # exit()
    print('encoding video...', end='')
    Writer = animation.writers['ffmpeg'](fps=fps)
    if slowmo:
        tag = '_slowmo'+tag
    ani.save(f'../data/vid2e/{name}/preproc{tag}.mp4', writer=Writer)
    print(f'Done.\n')

def main():
    filename = '../data/DAVIS240C-2021-03-17T10-49-45+0100-08360054-0.aedat4'   #DAVIS1
    filename = '../data/DAVIS240C-2018-10-12_pentagon0.5.aedat4'    #DAVIS2
    name = 'DAVIS2'
    # aedat4 file to numpy array
    # t, x, y, pol, img_dim = aedat2numpy(filename, norm_timestamps=True)

    # for visualizing kernel, always call with downsampling=True
    # the viz_kernel option downs't actually do anything to the data - no reshaping
    # viz_kernels = reshape_events(t, x, y, pol, img_dim, downsample=True, get_kernel_viz_params=True)
    # actually viz kernel areas in a video
    # events2vid_plot(t, x, y, pol, img_dim, fname=f'../data/{name}_nocrop_regions.mp4', name=name)

    # use reshape function to crop, but for visualization, first only get kernel viz parameters
    # viz_kernels = reshape_events(t, x, y, pol, img_dim, downsample=True, crop_to=(57,90,110,160), get_kernel_viz_params=True)
    # downsample is false because we want to visualize the kernel areas, this actually modifies the data to being cropped
    # t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, crop_to=(57,90,110,160), downsample=False)
    # actually viz kernel areas
    # events2vid_plot(t, x, y, pol, img_dim, fname=f'../data/{name}_cropped_regions.mp4', viz_kernels=viz_kernels, name=name)

    # finally downsampled data, cropping has already been applied so omitted for this reshape() call
    # t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, downsample=True)
    # events2vid_plot(t, x, y, pol, img_dim, fname=f'../data/{name}_downsampled.mp4', name=name)


    # x, y, t, pol = np.load('../data/vid2e/0.1_0.1_car.npy').T
    # img_dim = 180,240
    # name = 'car vid2e'
    # events2vid_plot(t, x, y, pol, img_dim, fname=f'../data/{name}.mp4', name=name)

    for which_data in range(len(names)):
    # for which_data in range(len(names)-1,-1,-1):
        if which_data not in (1,2):
            continue
        name = names[which_data]
        crop_to = crop_tos[which_data] 
        cut_to = cut_tos[which_data]
        tau = taus[which_data]
        thr = thrs[which_data]
        print('\n\ndataname: ', name)
        
        # read in a video file using skvideo
        video, fps, T, img_dim = video2numpy(name, cut_to)

        # get simulated events. Requires you to run the vid2events.py script first 
        # which needs specific esim_py env with opencv and stuff
        raw_events = from_numpy(f'../data/vid2e/{name}/simulated_events.npy')
        tr, xr, yr, polr = raw_events
        plot_isi(tr, xr, yr, polr, img_dim, name=name, tag='_original')
       
       
        # events2vid_plot(tr, xr, yr, polr, img_dim, name, T, fps, viz_reshape=crop_to)
        # plot_isi(tr, xr, yr, polr, img_dim)
        # continue

        # downsample events to 8 by 8
        td, xd, yd, pold, _ = reshape_events(tr, xr, yr, polr, img_dim, only_pol=None, 
                                             crop_to=crop_to, downsample=True)
        # td, xd, yd, pold = np.load(f'../data/vid2e/{name}/downs_cached.npy')
        plot_isi(td, xd, yd, pold, img_dim, name=name, tag='_downs')
        downs_events = np.stack([td, xd, yd, pold])
        np.save(f'../data/vid2e/{name}/downs_cached.npy', downs_events)

        # # try differnt filter settings for cleaning downsampled data
        # kernelsize = get_kernelsize(crop_to, img_dim)
        # find_filter_params(td, xd, yd, pold, name, kernelsize)
        
        # filter downsampled events
        tf, xf, yf, polf = filter_events(td, xd, yd, pold, name, tau, thr)
        filtd_events = np.stack([tf, xf, yf, polf])
        # check how the filtering went
        plot_isi(tf, xf, yf, polf, img_dim, name=name, tag='_downs_filtd')
        filtered_event_plot(td, xd, yd, pold, tf, xf, yf, polf, name=name)
        
        # make a video of the entire recording
        data = (video, raw_events, downs_events, filtd_events)
        plot_stimulus_preprocessing(data, fps, T, img_dim, name, viz_reshape=crop_to, 
                                    slowmo=True)


# general settings input data settings (imported by other scripts, that's why here)
tau, thr = .1, 60
names = 'falling_ball'  , 'large_ball_one', 'large_ball_two'
crop_tos = [(84, 180, 335, 450), (130, 123, 340, 408), 
            (130, 123, 335, 408)]
cut_tos = [(1,3), (2,5), (1,5)]

names =  'blink_slow', 'blink_medium', 'blink_fast', 'white_ball', 'black_ball'
Cps = [0.3, 0.3, 0.3, 0.3, 0.3]
Cns = [0.3, 0.3, 0.3, 0.3, 0.3]
refractory_periods = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
taus = [.6, .6, .6, .1, .1]
thrs = [20, 20, 20, 60, 60]
crop_tos = [(50, 85, 370, 540), (35, 110, 370, 540), (50, 85, 370, 540), (130, 225, 470, 620), (30, 80, 480, 440)]
cut_tos = [[0,6], [0,6], [0,6], [4.7,10.7], [28,35]]

if __name__ == '__main__':
    main()