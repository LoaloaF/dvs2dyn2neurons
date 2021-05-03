import time
import sys

import numpy as np

import samna  # samna is the python api to the dynapse chip
import samna.dynapse1 as dyn1

# change the path to '/home/class_NI2021/ctxctl_contrib' on zemo
# sys.path.insert(1, '/home/class_NI2021/ctxctl_contrib')
sys.path.insert(1, './ctxctl_contrib')
from Dynapse1Constants import *
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron
from process_aedat import dv2numpy, reshape_events

def numpyXY2FPGA_ID(x,y):
    serialize_yx = lambda coo: coo[0]*8 + coo[1]
    fpga_id = np.apply_along_axis(serialize_yx, 0, np.stack((y,x))).astype(int)
    return fpga_id

def get_dvs_input(aedat_file=None, crop_to=None):
    t, x, y, pol, img_dim = dv2numpy(aedat_file, norm_timestamps=True)
    t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, 
                                        crop_to=crop_to,
                                        downsample=True)
    fpga_ID = numpyXY2FPGA_ID(x, y)
    timestamps = t/1e6
    return fpga_ID, timestamps

def init_dynapse():
    device_name = "Board02"
    store = ut.open_dynapse1(device_name, gui=False, sender_port=12121, 
                             receiver_port=62321)
    model = getattr(store, device_name)

    # set initial (proper) parameters
    paramGroup = ut.gen_param_group_1core()
    model.update_parameter_group(paramGroup, 0, 0)
    return model, device_name, paramGroup


def setup_fpga_spikegen(model, fpga_ID, timestamps):
    # use 64 spikegens [0,64)
    post_chip = 0
    target_chips = np.full_like(fpga_ID, post_chip, dtype=int)
    isi_base = 900
    repeat_mode=False

    # get the fpga spike gen from Dynapse1Model
    fpga_spike_gen = model.get_fpga_spike_gen()
    # set up the fpga_spike_gen
    ut.set_fpga_spike_gen(fpga_spike_gen, timestamps, fpga_ID, target_chips, 
                          isi_base, repeat_mode)
    return fpga_spike_gen

def main():
    filename = '../data/DAVIS240C-2021-03-17T10-49-45+0100-08360054-0.aedat4'
    area = (57,90,110,160)

    fpga_ID, timestamps = get_dvs_input(filename, area)

    model, device_name, paramGroup = init_dynapse()
    # model = None

    spikegen = setup_fpga_spikegen(model, fpga_ID, timestamps)
    
    spikegen.start()


if __name__ == "__main__":
    main()
    print('Success!')