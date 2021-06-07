import esim_py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle

from create_stimulus import find_videofile


def get_frame_timestamps(video_file):
    cap = cv2.VideoCapture(video_file)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    T = 1/fps
    cap.release()

    return np.arange(0,nframes*T, T)

def video2events(name, Cp, Cn, refractory_period, cut_to=None):
    print('Simulating events from video.\n')
    video_file = find_videofile(f'/home/loaloa/Videos/{name}.mov')
    timestamps_file = f'/home/loaloa/Videos/{name}_timestamp.txt'
    dest_dir = f'/home/loaloa/gdrive/career/nsc_master/NI/project/data/vid2e/{name}'
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(timestamps_file):
        frame_ts = get_frame_timestamps(video_file)
        np.savetxt(timestamps_file, frame_ts)

    log_eps = 1e-3
    use_log = True
    esim = esim_py.EventSimulator(Cp, 
                                  Cn, 
                                  refractory_period, 
                                  log_eps, 
                                  use_log)
    events = esim.generateFromVideo(video_file, timestamps_file)

    # change order to t, x, y, pol (move t from 2 to 0) and also change to us
    events = np.concatenate([events[:,2:3]*1e6, events[:,(0,1,3)]], axis=1).astype(int)
    if cut_to is not None:
        # slice to correct section 
        cut_mask = (events[:,0]>cut_to[0]*1e6) & (events[:,0]<cut_to[1]*1e6)
        events = events[cut_mask]
    np.save(f'{dest_dir}/simulated_events.npy', events)

def main():
    from create_stimulus import names, cut_tos, Cps, Cns, refractory_periods
    
    for which_data in range(len(names)):
        name = names[which_data]
        cut_to = cut_tos[which_data]
        Cp = Cps[which_data]
        Cn = Cns[which_data]    
        refractory_period = refractory_periods[which_data]
        print('\n\ndataname: ', name)
        
        video2events(name, Cp, Cn, refractory_period, cut_to)

if __name__ == '__main__':
    main()
