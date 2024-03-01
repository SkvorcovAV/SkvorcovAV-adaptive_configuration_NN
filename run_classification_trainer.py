import torch
from core.models.morphic_neural_network import *
device = "cuda:0"
num_cl = 5
resize = (28,28)
#If sw are True, then only the structure of the neural network will change and there will be no change in the weight of connections.
# num_sinapse - Number of connections of each neuron in layer N to neurons in layer N+1. 
sw = False 
descriptions = "2DMorphicNetwork_ns_20_ww_2_nl_5_fs_28_28_lw_false_new_coord_syst"
net = SmallMorphicNetwork(input_size = resize,
                          num_classes = num_cl,
                          num_sinapse = 20,
                          window_width = 2,
                          num_layers = 5,
                          feauters_size  = resize,
                          sw = sw)

net = FlatMorphicNetwork(input_size = resize,
                         num_classes = num_cl,
                         num_sinapse = 20,
                         window_width = 4,
                         num_layers = 5,
                         num_future  = 800,
                         sw = sw)

optimizer = morphological_net_optimizer(net)


