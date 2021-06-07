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

def main():
    from create_stimulus import names, cut_tos
    
    for which_data in range(len(names)):
        name = names[which_data]
        cut_to = cut_tos[which_data]
        print('\n\ndataname: ', name)
        
        aedat2numpy(name, Cp, Cn, refractory_period, cut_to)

if __name__ == '__main__':
    main()
