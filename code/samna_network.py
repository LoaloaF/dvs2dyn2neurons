import time
import sys

import numpy as np
from random import randint
import pickle

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

CHIP_ID = 0
FPGA_NEURONS_CORE = 0
N_WTA_NEURONS = 64
WTA_NEURONS_CORE = 1
N_INH_NEURONS = 64
INH_NEURONS_CORE = 2
N_STIM_NEURONS = 64
STIM_NEURONS_CORE = 3
PIXEL_IDS = np.arange(N_WTA_NEURONS)

def get_dvs_input(aedat_file=None, crop_to=None, only_pol=None):
    t, x, y, pol, img_dim = dv2numpy(aedat_file, norm_timestamps=True)
    t, x, y, pol, img_dim = reshape_events(t, x, y, pol, img_dim, 
                                        crop_to=crop_to,
                                        downsample=True, only_pol=only_pol)
    
    serialize_yx = lambda coo: coo[0]*8 + coo[1]
    fpga_ID = np.apply_along_axis(serialize_yx, 0, np.stack((y,x))).astype(int)
    timestamps = t/1e6
    return fpga_ID, timestamps

def dvs2stimulus(model, filename, crop_to=None, only_pol=None):
    fpga_ID, timestamps = get_dvs_input(filename, crop_to, only_pol)
    post_chip = 0
    target_chips = np.full_like(fpga_ID, post_chip, dtype=int)
    isi_base = 900
    repeat_mode=False

    fpga_spike_gen = model.get_fpga_spike_gen()
    ut.set_fpga_spike_gen(fpga_spike_gen, timestamps, fpga_ID, target_chips, 
                          isi_base, repeat_mode)
    return fpga_spike_gen, timestamps[-1]

def init_dynapse():
    device_name = "Board02"
    port1, port2 = randint(11111,99999), randint(11111,99999)
    store = ut.open_dynapse1(device_name, gui=False, sender_port=port1, 
                             receiver_port=port2)
    model = getattr(store, device_name)

    # set initial (proper) parameters
    paramGroup = ut.gen_param_group_1core()
    for core in [FPGA_NEURONS_CORE, WTA_NEURONS_CORE, INH_NEURONS_CORE, STIM_NEURONS_CORE]:
        for chip in range(4):
            model.update_parameter_group(paramGroup, chip, core)

    return store, model, device_name, paramGroup

def build_network(model):
    net_gen = n.NetworkGenerator()
    # create all the neurons
    fpga_inputs, wta_neurons, inh_neurons, stim_neurons = [], [], [], []
    for pid in PIXEL_IDS:
        wta_neurons.append(Neuron(CHIP_ID, WTA_NEURONS_CORE, pid))
        fpga_inputs.append(Neuron(CHIP_ID, WTA_NEURONS_CORE, pid, is_spike_gen=True))
        inh_neurons.append(Neuron(CHIP_ID, INH_NEURONS_CORE, pid))
        stim_neurons.append(Neuron(CHIP_ID, STIM_NEURONS_CORE, pid))

    # gaussian averaging over WTA pop?
    # create one-to-one connections between 8x8 fpga input, and WTA neurons
    for pid in PIXEL_IDS:
        net_gen.add_connection(fpga_inputs[pid], wta_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        net_gen.add_connection(wta_neurons[pid], inh_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        for nid, wta_n in enumerate(wta_neurons):
            if nid != pid:
                net_gen.add_connection(inh_neurons[pid], wta_n, dyn1.Dynapse1SynType.GABA_B)
        net_gen.add_connection(wta_neurons[pid], stim_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        
    net_gen.print_network()
    new_config = net_gen.make_dynapse1_configuration()
    model.apply_configuration(new_config)


def get_monitors(model, to_monitor=['FPGA', 'WTA', 'INH', 'STIM']):
    monitors = {key: {} for key in to_monitor}
    for mon in monitors:
        print('\n\n', mon, ' global neuron IDS:')
        if mon == 'FPGA':
            core = FPGA_NEURONS_CORE
        elif mon == 'WTA':
            core = WTA_NEURONS_CORE
        elif mon == 'INH':
            core = INH_NEURONS_CORE
        elif mon == 'STIM':
            core = STIM_NEURONS_CORE
        global_ids = [ut.get_global_id(CHIP_ID, core, nid) for nid in PIXEL_IDS]
        print(global_ids)
        graph, filter_node, sink_node = ut.create_neuron_select_graph(model, global_ids)
        filter_node.set_neurons(global_ids)

        monitors[mon]['graph'] = graph
        monitors[mon]['filter_node'] = filter_node
        monitors[mon]['sink_node'] = sink_node
    return monitors

def run_network(monitors, duration, stimulus=None):
    [monitor['graph'].start() for monitor in monitors.values()]
    if stimulus:
        stimulus.start()

    [monitor['sink_node'].get_buf() for monitor in monitors.values()]
    time.sleep(duration)
    for monitor in monitors.values():
        monitor['events'] = monitor['sink_node'].get_buf()

    if stimulus:
        stimulus.stop()
    [monitor['graph'].stop() for monitor in monitors.values()]

def process_output(monitors, name=''):
    events_numpy = {}
    for mon_name, monitor in monitors.items():
        print(f'\n\n\n\n{mon_name}: n events: {len(monitor["events"])}\n')
        ev_npy = [(ev.neuron_id, ev.timestamp) for ev in monitor['events']]
        if ev_npy:
            events_numpy[mon_name] = np.stack(ev_npy)

    pickle.dump(events_numpy, open(f'../data/events_{name}.pkl', 'wb'))







        
def main():
    store, model, device_name, paramGroup = init_dynapse()

    # build_network(model)

    # filename = '../data/DAVIS240C-2021-03-17T10-49-45+0100-08360054-0.aedat4'
    # area = (57,90,110,160)
    # only_pol = 1
    # stimulus, duration = dvs2stimulus(model, filename, crop_to=area, only_pol=only_pol)

    duration = 20
    monitors = get_monitors(model)
    run_network(monitors, duration)
    process_output(monitors, name='DAVIS1')

    # print(monitors['FPGA']['events'][0].neuron_id)
    # # print(monitors['FPGA']['events'][0].isi)
    # print(monitors['STIM']['events'][0].neuron_id)
    # print(monitors['STIM']['events'][0].timestamp)

    ut.close_dynapse1(store, device_name)

if __name__ == "__main__":
    main()
    print('Success!')