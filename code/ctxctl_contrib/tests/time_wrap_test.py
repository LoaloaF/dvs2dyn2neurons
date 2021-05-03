import samna
import samna.dynapse1 as dyn1
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron
from Dynapse1Constants import *
import time

def gen_param_group_1core():
    paramGroup = dyn1.Dynapse1ParameterGroup()
    # THR
    # ok
    paramGroup.param_map["IF_THR_N"].coarse_value = 5
    paramGroup.param_map["IF_THR_N"].fine_value = 80

    # refactory period
    paramGroup.param_map["IF_RFR_N"].coarse_value = 4
    paramGroup.param_map["IF_RFR_N"].fine_value = 128

    # leakage
    paramGroup.param_map["IF_TAU1_N"].coarse_value = 4
    paramGroup.param_map["IF_TAU1_N"].fine_value = 80

    paramGroup.param_map["IF_TAU2_N"].coarse_value = 7
    paramGroup.param_map["IF_TAU2_N"].fine_value = 255

    paramGroup.param_map["IF_DC_P"].coarse_value = 0
    paramGroup.param_map["IF_DC_P"].fine_value = 0

    paramGroup.param_map["NPDPIE_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_F_P"].fine_value = 80

    paramGroup.param_map["NPDPIE_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_F_P"].fine_value = 80

    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 7
    paramGroup.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 80

    paramGroup.param_map["NPDPIE_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_TAU_S_P"].fine_value = 80

    paramGroup.param_map["NPDPIE_THR_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPIE_THR_S_P"].fine_value = 80

    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 4
    paramGroup.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 80

    paramGroup.param_map["IF_NMDA_N"].coarse_value = 0
    paramGroup.param_map["IF_NMDA_N"].fine_value = 0

    paramGroup.param_map["NPDPII_TAU_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_F_P"].fine_value = 80

    paramGroup.param_map["NPDPII_THR_F_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_THR_F_P"].fine_value = 80

    paramGroup.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_INH_F_N"].fine_value = 0

    paramGroup.param_map["NPDPII_TAU_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_TAU_S_P"].fine_value = 80

    paramGroup.param_map["NPDPII_THR_S_P"].coarse_value = 4
    paramGroup.param_map["NPDPII_THR_S_P"].fine_value = 80

    paramGroup.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 0
    paramGroup.param_map["PS_WEIGHT_INH_S_N"].fine_value = 0

    paramGroup.param_map["IF_AHTAU_N"].coarse_value = 4
    paramGroup.param_map["IF_AHTAU_N"].fine_value = 80

    paramGroup.param_map["IF_AHTHR_N"].coarse_value = 0
    paramGroup.param_map["IF_AHTHR_N"].fine_value = 0

    paramGroup.param_map["IF_AHW_P"].coarse_value = 0
    paramGroup.param_map["IF_AHW_P"].fine_value = 0

    paramGroup.param_map["IF_CASC_N"].coarse_value = 0
    paramGroup.param_map["IF_CASC_N"].fine_value = 0

    paramGroup.param_map["PULSE_PWLK_P"].coarse_value = 4
    paramGroup.param_map["PULSE_PWLK_P"].fine_value = 106

    paramGroup.param_map["R2R_P"].coarse_value = 3
    paramGroup.param_map["R2R_P"].fine_value = 85

    paramGroup.param_map["IF_BUF_P"].coarse_value = 3
    paramGroup.param_map["IF_BUF_P"].fine_value = 80

    return paramGroup

# open DYNAP-SE1 board to get Dynapse1Model
device_name = "dynapse1"
store, gui_process = ut.open_dynapse1(device_name)
model = getattr(store, device_name)

# get Dynapse1 api from the model
api = model.get_dynapse1_api()

serial_number = ut.get_serial_number(store, device_name)
print(device_name, "serial number is", serial_number)

# monitor neuron using oscilloscope
print("Monitor neuron 123 in chip 1")
api.monitor_neuron(1, 123)

# ------------------- build network -------------------
net_gen = n.NetworkGenerator()

spikegen_ids = [(0,2,50)]
spikegens = []
for spikegen_id in spikegen_ids:
    spikegens.append(Neuron(spikegen_id[0],spikegen_id[1],spikegen_id[2],True))

chip = 1
neuron_ids = [(chip,0,20), (0,0,36), (0,2,60), (1,1,60), (2,1,107), (2,3,152)]
neurons = []
for nid in neuron_ids:
    neurons.append(Neuron(nid[0],nid[1],nid[2]))

# connect spikeGen to neuron0
w = 1
for i in range(w):
    net_gen.add_connection(spikegens[0], neurons[0], dyn1.Dynapse1SynType.AMPA)

# connect neuron0 to other neurons
net_gen.add_connection(neurons[0], neurons[1], dyn1.Dynapse1SynType.AMPA)
net_gen.add_connection(neurons[0], neurons[2], dyn1.Dynapse1SynType.NMDA)
net_gen.add_connection(neurons[0], neurons[3], dyn1.Dynapse1SynType.GABA_A)
net_gen.add_connection(neurons[1], neurons[4], dyn1.Dynapse1SynType.GABA_B)
net_gen.add_connection(neurons[2], neurons[5], dyn1.Dynapse1SynType.GABA_B)

# print the network so you can double check (optional)
net_gen.print_network()

# make a dynapse1config using the network
new_config = net_gen.make_dynapse1_configuration()

# apply the configuration
model.apply_configuration(new_config)
# ------------------- build network -------------------

# set spike generators
global_poisson_gen_ids = ut.get_global_id_list(spikegen_ids)
rate = 200
poisson_gen = model.get_poisson_gen()
poisson_gen.set_chip_id(chip)
poisson_gen.write_poisson_rate_hz(global_poisson_gen_ids[0], rate)

# check the configuration...
global_ids = ut.get_global_id_list(neuron_ids)
config = model.get_configuration()
for i in range(len(global_ids)):
    nid = global_ids[i]
    neuron = ut.get_neuron_from_config(config, nid)
    print("------------Neuron", neuron_ids[i],"------------")
    print("Cams:")
    ut.print_neuron_synapses(neuron, range(12))
    print("Srams:")
    ut.print_neuron_destinations(neuron)

paramGroup = gen_param_group_1core()
for chip in range(4):
    for core in range(4):
        model.update_parameter_group(paramGroup, chip, core)

        # make the neurons fire to get the timewrapEvent
        model.update_single_parameter(dyn1.Dynapse1Parameter("IF_DC_P",5,30), chip, core)

# # save parameters
ut.save_parameters2txt_file(model.get_configuration(), './current_parameters.txt')
ut.save_parameters2json_file(model.get_configuration(), './current_parameters.json')

ut.set_parameters_in_json_file(model, './current_parameters.json')


# start the poisson gen
poisson_gen.start()

# run some experiments...
time.sleep(2)

# stop the poisson gen
# must stop otherwise you have to unplug next time you run the program
poisson_gen.stop()

poisson_gen.start()

monitored_neurons = global_ids[:1]
graph, filter_node, sink_node = ut.create_neuron_select_graph(model, monitored_neurons)
graph.start()

graph_wrap, sink_node_wrap = ut.get_time_wrap_events(model)
graph_wrap.start()

t0 = time.time()
sink_node.get_buf() # clear the buffer
while True:
    time_wrap_events = sink_node_wrap.get_buf()
    for evt in time_wrap_events:
        print(evt.timestamp)

    spikes = sink_node.get_buf()
    if len(spikes):
        print("systime sec",int(time.time()-t0), len(spikes),"spikes. Last spike ts:",spikes[-1].timestamp)
    del spikes

    time.sleep(120) # 2 min


graph.stop()
graph_wrap.stop()

poisson_gen.stop()

# close Dynapse1
ut.close_dynapse1(store, device_name, gui_process)