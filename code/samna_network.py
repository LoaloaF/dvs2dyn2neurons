import time
import sys

import numpy as np
from random import randint
import pickle

import samna  # samna is the python api to the dynapse chip
import samna.dynapse1 as dyn1

sys.path.insert(1, './ctxctl_contrib')
from Dynapse1Constants import *
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron

from create_stimulus import from_numpy, serialize_nids, unserialize_nids

def neighb_pix(pid, get='all', level=1, drop_corners=False):
    # grid = np.arange(64).reshape(8,8).astype(int)
    x_pid, y_pid = unserialize_nids(pid)
    coos = []
    for dy in range(-level,level+1):
        for dx in range(-level,level+1):
            y_coo, x_coo = y_pid+dy ,x_pid+dx
            yedge = abs(dy)==level
            xedge = abs(dx)==level
            if (yedge or xedge) and (0<=y_coo<8) and (0<=x_coo<8):
                if drop_corners and yedge and xedge:
                    continue
                coos.append([y_coo,x_coo])
    coos = np.array(coos).T
    if get == 'top':
        mask = coos[0] == y_pid-level
    elif get == 'bottom':
        mask = coos[0] == y_pid+level
    elif get == 'left':
        mask = coos[1] == x_pid-level
    elif get == 'right':
        mask = coos[1] == x_pid+level

    else:
        mask = np.ones_like(coos[0], bool)

    y_coo, x_coo = coos
    y_coo, x_coo = y_coo[mask], x_coo[mask]
    # grid[y_coo, x_coo] = 0
    # print(grid)
    return serialize_nids(x_coo, y_coo)


def get_dvs_input(datafile, sample=None, timeslice=None):
    t, x, y, pol = from_numpy(datafile, sample=sample, timeslice=timeslice)
    fpga_ID = serialize_nids(x, y)
    fpga_ID = np.where(pol==1, fpga_ID, fpga_ID+64)
    timestamps = t/1e6
    return fpga_ID, timestamps

def dvs2stimulus(model, filename, sample=None, timeslice=None, get_rawDVS_input=True):
    fpga_ID, timestamps = get_dvs_input(filename, sample, timeslice)
    target_chips = np.full_like(fpga_ID, CHIP_ID, dtype=int)
    isi_base = 900
    repeat_mode = False

    fpga_spike_gen = model.get_fpga_spike_gen()
    ut.set_fpga_spike_gen(fpga_spike_gen, timestamps, fpga_ID, target_chips, 
                          isi_base, repeat_mode)
    if get_rawDVS_input:
        return fpga_spike_gen, timestamps[-1], fpga_ID, (timestamps*1e6).astype(int)
    else:
        return fpga_spike_gen, timestamps[-1]

def init_dynapse():
    device_name = "Board02"
    port1, port2 = randint(11111,99999), randint(11111,99999)
    store = ut.open_dynapse1(device_name, gui=False, sender_port=port1, 
                             receiver_port=port2)
    model = getattr(store, device_name)

    # set initial (proper) parameters
    paramGroup = ut.gen_param_group_1core()
    for core in [WTA_NEURONS_CORE, INH_NEURONS_CORE, STIM_NEURONS_CORE]:
        for chip in range(4):
            model.update_parameter_group(paramGroup, chip, core)

    return store, model, device_name, paramGroup

def get_WTA_connectivity(pid, on_or_off):
    nids = neighb_pix(pid)
    nids_top = neighb_pix(pid, get='top', level=2, drop_corners=True)
    nids_bottom = neighb_pix(pid, get='bottom', level=2, drop_corners=True)
    
    nids = np.concatenate([nids_top,nids,nids_bottom]).flatten()
    nids = sorted(list(nids) + [pid])

    if on_or_off == 'on':
        n_conns = np.array(([ 0, 2, 0],
                           [ 1, 4, 1], 
                           [ 0, 0, 0], 
                           [-1,-4,-1], 
                           [ 0,-2, 0])) 
    elif on_or_off == 'off':
        n_conns = np.array(([ 0, 2, 0],
                           [ 1, 4, 1], 
                           [ 0,-4, 0], 
                           [-1,-2,-1], 
                           [ 0, 0, 0])) 
    n_conns = n_conns.flatten()
    return nids, n_conns
 

def build_network(model):
    net_gen = n.NetworkGenerator()
    # create all the neurons
    fpga_inputs, wta_neurons, inh_neurons, stim_neurons = [], [], [], []
    for pid in PIXEL_IDS:
        # spikegen needs its own population?
        fpga_inputs.append(Neuron(CHIP_ID, FPGA_NEURONS_CORE, pid, is_spike_gen=True))
        wta_neurons.append(Neuron(CHIP_ID, WTA_NEURONS_CORE, pid))
        inh_neurons.append(Neuron(CHIP_ID, INH_NEURONS_CORE, pid))
        stim_neurons.append(Neuron(CHIP_ID, STIM_NEURONS_CORE, pid))
    for pid in PIXEL_IDS:
        fpga_inputs.append(Neuron(CHIP_ID, FPGA_NEURONS_CORE, pid+64, is_spike_gen=True))
    
    # gaussian averaging over WTA pop?
    # create one-to-one connections between 8x8 fpga input, and WTA neurons
    for pid in PIXEL_IDS:
        on_fpga_id, off_fpga_id = pid, pid+64
        # net_gen.add_connection(fpga_inputs[on_fpga_id], wta_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        # net_gen.add_connection(fpga_inputs[off_fpga_id], wta_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        
        select_syn = lambda conn_n: dyn1.Dynapse1SynType.AMPA if conn_n >0 else dyn1.Dynapse1SynType.GABA_B
        # on events
        wta_nids, n_conns = get_WTA_connectivity(pid, 'on')
        for wta_nid, n_conn in zip(wta_nids, n_conns):
            [net_gen.add_connection(fpga_inputs[on_fpga_id], wta_neurons[wta_nid], select_syn(cn))
             for cn in range(n_conn,0)]
        
        # off events
        wta_ns, n_conns = get_WTA_connectivity(pid, 'off')
        for wta_nid, n_conn in zip(wta_nids, n_conns):
            [net_gen.add_connection(fpga_inputs[off_fpga_id], wta_neurons[wta_nid], select_syn(cn))
             for cn in range(n_conn,0)]
        
        # WTA to inhibitory exc connection
        net_gen.add_connection(wta_neurons[pid], inh_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        
        # # inhibitory feedback from one INH neuron to all WTA neurons
        # pid_neighbours = neighb_pix(pid)
        # for nid, wta_n in enumerate(wta_neurons):
        #     if nid != pid and nid not in pid_neighbours:
        #         net_gen.add_connection(inh_neurons[pid], wta_n, dyn1.Dynapse1SynType.GABA_B)

        # inhibitory-inhibitory lateral exc connection
        for inh_n in pid_neighbours[::2]:
            net_gen.add_connection(inh_neurons[pid], inh_neurons[inh_n], dyn1.Dynapse1SynType.AMPA)

        # one 2 one WTA to stimulating pop exc connection
        net_gen.add_connection(wta_neurons[pid], stim_neurons[pid], dyn1.Dynapse1SynType.AMPA)
        
    net_gen.print_network()
    new_config = net_gen.make_dynapse1_configuration()
    model.apply_configuration(new_config)

def set_parameters(model, paramGroup=None):
    pass
    # turn down lateral exc connections in inhibitory population
    # param = dyn1.Dynapse1Parameter("PS_WEIGHT_EXC_F_N", 5, 200)
    # model.update_single_parameter(param, CHIP_ID, INH_NEURONS_CORE)

def get_monitors(model, to_monitor=['WTA', 'INH', 'STIM']):
    monitors = {key: {} for key in to_monitor}
    for mon in monitors:
        if mon == 'WTA':
            core = WTA_NEURONS_CORE
        elif mon == 'INH':
            core = INH_NEURONS_CORE
        elif mon == 'STIM':
            core = STIM_NEURONS_CORE
        global_ids = [ut.get_global_id(CHIP_ID, core, nid) for nid in PIXEL_IDS]
        graph, filter_node, sink_node = ut.create_neuron_select_graph(model, global_ids)
        # filter_node.set_neurons(global_ids)   # necessary?

        monitors[mon]['graph'] = graph
        monitors[mon]['filter_node'] = filter_node
        monitors[mon]['sink_node'] = sink_node
    return monitors

def run_network(monitors, duration, stimulus):
    print(f'\nRunning for {duration}s')
    [monitor['graph'].start() for monitor in monitors.values()]
    stimulus.start()

    [monitor['sink_node'].get_buf() for monitor in monitors.values()]
    time.sleep(duration)
    for monitor in monitors.values():
        monitor['events'] = monitor['sink_node'].get_buf()

    stimulus.stop()
    [monitor['graph'].stop() for monitor in monitors.values()]

def process_output(monitors, name, DVS_input=None):
    events_numpy = {}
    if DVS_input:
        fpga_ID, timestamps = DVS_input
        # print(fpga_ID[::400])
        pol = np.where(fpga_ID-64 <0, 1, -1)
        fpga_ID = np.where(fpga_ID>63, fpga_ID-64, fpga_ID)
        x, y = unserialize_nids(fpga_ID)
        # print(fpga_ID[::400])
        # print(pol[::400])
        events_numpy['DVS'] = np.stack((timestamps, x, y, pol)).T

    for mon_name, monitor in monitors.items():
        print(f'{mon_name} n events: {len(monitor["events"])}')
        ev_npy = [(ev.timestamp, *unserialize_nids(ev.neuron_id))
                  for ev in monitor['events']]
        if ev_npy:
            events_numpy[mon_name] = np.stack(ev_npy)
    pickle.dump(events_numpy, open(f'../data/vid2e/{name}/dyn_output.pkl', 'wb'))



CHIP_ID = 0
FPGA_NEURONS_CORE = 0
N_WTA_NEURONS = 64
WTA_NEURONS_CORE = 1
N_INH_NEURONS = 64
INH_NEURONS_CORE = 2
N_STIM_NEURONS = 64
STIM_NEURONS_CORE = 3
PIXEL_IDS = np.arange(N_WTA_NEURONS)

def main():
    from create_stimulus import names

    which_data = 4  # black_ball
    name = names[which_data]
    store, model, device_name, paramGroup = init_dynapse()
    # model = None
    
    filename = f'../data/vid2e/{name}/DVS_stimulus.npy'
    stimulus, duration, fpga_ID, timestamps = dvs2stimulus(model, filename, 32500, (0,3e6))

    build_network(model)

    monitors = get_monitors(model)
    set_parameters(model, paramGroup)
    run_network(monitors, duration, stimulus)
    process_output(monitors, name=name, DVS_input=(fpga_ID, timestamps))

    ut.close_dynapse1(store, device_name)

if __name__ == "__main__":
    main()
    print('Success!\n')