import samna.dynapse1 as dyn1
import Dynapse1Utils as ut
import NetworkGenerator as n
from NetworkGenerator import Neuron
from Dynapse1Constants import *
import time
import samna
import json

# read the samna info from json file written by py_server.py
with open('./samna_info.json') as f:
    data = json.load(f)
print(data)
sender_port = data["sender_port"]
receiver_port = data["receiver_port"]
samna_node_id = data["samna_node_id"]
device_name = data["device_name"]
python_node_id = data["python_node_id"]

python_node_id += 3

# setup the python interpreter node
samna.setup_local_node(receiver_port, sender_port, python_node_id)
# open a connection to device_node, i.e. the store. Must open the device_node here
samna.open_remote_node(samna_node_id, "device_node")
store = samna.device_node
model = getattr(store, device_name)

# get Dynapse1 api from the model
api = model.get_dynapse1_api()

# ------------------- build network -------------------
net_gen = n.NetworkGenerator()

chip = 0
core = 3
spikegen_ids = [(0,core,66,True)]
spikegens = []
for spikegen_id in spikegen_ids:
    spikegens.append(Neuron(spikegen_id[0],spikegen_id[1],spikegen_id[2],spikegen_id[3]))
neuron_ids = [(chip,core,10), (chip,core,25), (chip,core,31), (chip,core,65), (chip,core,107), (chip,core,152)]
neurons = []
for nid in neuron_ids:
    neurons.append(Neuron(nid[0],nid[1],nid[2]))

net_gen.add_connection(spikegens[0], neurons[0], dyn1.Dynapse1SynType.AMPA)
net_gen.add_connection(neurons[0], neurons[1], dyn1.Dynapse1SynType.AMPA)
net_gen.add_connection(neurons[0], neurons[2], dyn1.Dynapse1SynType.NMDA)
net_gen.add_connection(neurons[0], neurons[3], dyn1.Dynapse1SynType.GABA_A)
net_gen.add_connection(neurons[1], neurons[4], dyn1.Dynapse1SynType.GABA_B)
net_gen.add_connection(neurons[2], neurons[5], dyn1.Dynapse1SynType.GABA_B)

# print the network so you can double check (optional)
net_gen.print_network()

# make a dynapse1config using the network
new_config = net_gen.make_dynapse1_configuration_in_core(chip, core)

# apply the configuration
model.apply_configuration_by_core(new_config, chip, core)
# ------------------- build network -------------------

# check the configuration...
global_ids = ut.get_global_id_list(neuron_ids)
config = model.get_configuration()
for i in range(len(global_ids)):
        nid = global_ids[i]
        neuron = ut.get_neuron_from_config(config, nid)
        print("------------Neuron", neuron_ids[i],"------------")
        print("Cams:")
        ut.print_neuron_synapses(neuron, range(2))
        print("Srams:")
        ut.print_neuron_destinations(neuron)

# set spike generators (0,1,66)
poisson_gen_id = ut.get_global_id(0,1,66)
rate = 200
poisson_gen = model.get_poisson_gen()
poisson_gen.set_chip_id(0)
poisson_gen.write_poisson_rate_hz(poisson_gen_id, rate) # neuron20 on chip 0 core 0

# set parameters
ut.set_parameters_in_txt_file(model, './parameters.txt')

# save parameters
ut.save_parameters2txt_file(model.get_configuration(), './current_parameters_sharing.txt')

# start the poisson gen
poisson_gen.start()


poisson_gen.stop()

while True:
    pass