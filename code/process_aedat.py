import sys
import os

from dv import AedatFile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from skimage.util import img_as_ubyte

# # for more complicated colouring 
# from matplotlib import cm
# from matplotlib import colors
# cmap = cm.get_cmap('seismic')
# vmin, vmax = -1, 1
# norm_col = colors.Normalize(vmin=vmin, vmax=vmax)

def dv2numpy(datafile, norm_timestamps=True):
    with AedatFile(datafile) as f:
        img_dim = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])

        timestamps = np.array(events['timestamp'])
        x = np.array(events['x'])
        y = np.array(events['y'])
        polarities = np.array(events['polarity'])

        if norm_timestamps:
            timestamps -= timestamps[0]
        return timestamps, x, y, polarities, img_dim


def reshape_events(t, x, y, pol, img_dim, downsample=False, crop_to=None, only_pol=None, get_kernel_viz_params=False):
    # limit the specific polarity, 0 or 1
    if only_pol is not None:
        idx_pol = np.where(pol==only_pol)[0]
        # slice to those
        t, y, x, pol = t[idx_pol], y[idx_pol], x[idx_pol], pol[idx_pol]
    
    # slice to some field of view
    if crop_to is not None:
        ymin, ymax, xmin, xmax = crop_to
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
    return t, x, y, pol, img_dim


def numpy2vid(t, x, y, pol, img_dim, T=16000, fps=None, fname='../data/output.mp4', 
              viz_kernels=None, name=''):
    # empty image
    nframes = t[-1] // T
    if fps is None:
        # set fps to match actual raw recording length
        fps = int(1*1e6/T)
        
    ysize, xsize = img_dim
    # empty rgb frame sequence, actually not used anymore
    video = np.zeros((nframes, ysize, xsize, 3), np.ubyte)

    # if kernel areas are drawn, unpack the viz_kernel list, pad video + x & y
    if viz_kernels is not None:
        ypad, xpad, ykernels, xkernels = viz_kernels
        video = np.zeros((nframes, ypad[0]+ysize+ypad[1], xpad[0]+xsize+xpad[1], 3), np.ubyte)
        ysize, xsize = video.shape[1:-1]
        y += ypad[0]
        x += xpad[0]
    # video[:] = img_as_ubyte(cmap(.5))[:-1]
    video[:] = (240,240,240) # whiteish

    bin_range = (0,0)
    # this collects a list for each frame containing the heatmap and a text label artist
    mpl_frames = []
    # the video is made with matplotlibs animations API
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_title(name, pad=60, fontsize=15)
    ax.axis('off')
    if not 1 in pol:
        polarities = '[only on events]'
    elif not 0 in pol:
        polarities = '[only off events]'
    else:
        polarities = '[both on & off events]'

    for frame in range(nframes):
        if not frame % 100:
            print(f'frame {frame}/{nframes}...', end='')
        # move the bin boundary by the periodlength
        bin_range = bin_range[1], bin_range[1]+T
        # slice to timepoints that are in the timebin
        idx_in_bin = np.argwhere((bin_range[0] <= t) & (t < bin_range[1]))[:,0]
        
        # draw events if any for that timerange/ frame
        if idx_in_bin.any():
            # maybe useful later when not polarity not just between 0,1:
            # frame_pol = 2*pol[idx_in_bin] -1
            # frame_pol_color = img_as_ubyte(cmap(norm_col(frame_pol)))[:,:-1]
            # for now, just color off events blue, on events red 
            frame_pol_color = [(255,0,0) if p == 1 else (0,0,255) for p in pol[idx_in_bin]]
            # slice coordinates to timebins, set their color in the video
            coo = np.stack((y[idx_in_bin], x[idx_in_bin]))
            video[frame, coo[0], coo[1], :] = frame_pol_color

        # draw heatmap
        frame = ax.imshow(video[frame], animated=True)
        # labelling
        lbl = f't: {bin_range[1]/1e6:.2f} s, T: {T//1e3} ms, FPS: {fps}, polarities: {polarities}'
        text = ax.text(0,1.05, lbl, transform=ax.transAxes, fontsize=13)
        # # spines
        # ax.vlines((-.5, xsize-.5), (-.5,-.5),(ysize-.5,ysize-.5), color='k', linewidth=.5),
        # ax.hlines((-.5, ysize-.5), (-.5,-.5),(xsize-.5,xsize-.5), color='k', linewidth=.5)

        # draw kernel spines
        if viz_kernels is not None:
            x_coos = np.arange(0, 8*xkernels, xkernels)[1:]
            y_coos = np.arange(0, 8*ykernels, ykernels)[1:]
            ax.vlines(x_coos, np.full_like(x_coos, -.5), np.full_like(x_coos, ysize-.5), color='k', linewidth=.5),
            ax.hlines(y_coos, np.full_like(y_coos, -.5), np.full_like(y_coos, xsize-.5), color='k', linewidth=.5),

        mpl_frames.append([frame, text])
    ani = animation.ArtistAnimation(fig, mpl_frames, interval=5, blit=True)
    print('encoding video...', end='')
    Writer = animation.writers['ffmpeg'](fps=fps)
    ani.save(fname, writer=Writer)
    print(f'Saved: {fname}')

def main():
    filename = '../data/DAVIS240C-2021-03-17T10-49-45+0100-08360054-0.aedat4'   #DAVIS1
    filename = '../data/DAVIS240C-2018-10-12_pentagon0.5.aedat4'    #DAVIS2
    name = 'DAVIS2'
    # aedat4 file to numpy array
    t, x, y, pol, img_dim = dv2numpy(filename, norm_timestamps=True)

    # for visualizing kernel, always call with downsampling=True
    # the viz_kernel option downs't actually do anything to the data - no reshaping
    # viz_kernels = reshape_events(t, x, y, pol, img_dim, downsample=True, get_kernel_viz_params=True)
    # actually viz kernel areas in a video
    # numpy2vid(t, x, y, pol, img_dim, fname=f'../data/{name}_nocrop_regions.mp4', viz_kernels=viz_kernels, name=name)

    # use reshape function to crop, but for visualization, first only get kernel viz parameters
    # viz_kernels = reshape_events(t, x, y, pol, img_dim, downsample=True, crop_to=(57,90,110,160), get_kernel_viz_params=True)
    # downsample is false because we want to visualize the kernel areas, this actually modifies the data to being cropped
    # t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, crop_to=(57,90,110,160), downsample=False)
    # actually viz kernel areas
    # numpy2vid(t, x, y, pol, img_dim, fname=f'../data/{name}_cropped_regions.mp4', viz_kernels=viz_kernels, name=name)

    # finally downsampled data, cropping has already been applied so omitted for this reshape() call
    t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, downsample=True)
    numpy2vid(t, x, y, pol, img_dim, fname=f'../data/{name}_downsampled.mp4', name=name)

if __name__ == '__main__':
    main()