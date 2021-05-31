from dv import AedatFile
import numpy as np

def aedat2numpy(datafile, norm_timestamps=True):
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