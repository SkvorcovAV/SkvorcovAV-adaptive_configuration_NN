import os
import numpy as np
import torch
from torch import nn, optim
import itertools
import math
import csv


class morphological_net_optimizer(optim.Optimizer):
    def __init__(self,
                 net,
                 learning_rate_synapses_weight=0.00001,
                 learning_rate_synapses_coordinates = 1):
        
        self.max_move = net.ww
        params = list(net.named_parameters())
        
        weight = [param for name, param in params if "synapses_weight" in name]
        coordinates = [param for name, param in params if 'coordinates' in name]
        self.coordinates_names = [name for name, param in params if 'coordinates' in name]
        self.count = -1
        self.velocricys = []
        
        for tenzor in coordinates:
            new_tenzor = torch.zeros(tenzor.size()).to(tenzor.device)
            self.velocricys.append(new_tenzor)
            
        limit = []
        
        for name, param in params:
            if 'output_size' in name:
                for i in range(len(param)):
                    limit.append(param[i])            
        
        param_groups = [{'params': coordinates,
                         'limit': limit,
                         'learning_rate': learning_rate_synapses_coordinates,
                         "group_name":"synapses_coordinates"},
                        
                        {'params': weight,
                         'learning_rate': learning_rate_synapses_weight,
                         "group_name":"synapses_weight"}]
        defaults = {}
        super(morphological_net_optimizer, self).__init__(param_groups, defaults)
        
    
    def step(self):
        for group in self.param_groups:
            if group["group_name"] == "synapses_weight":
                for param in group['params']:
                    if param.grad is None:
                        continue
                    grad = param.grad.data
                    lr = group['learning_rate']
                    param.data -= lr * grad
                
            else:
                
                self.count+=1
                for param, limit, name in zip(group['params'], group['limit'], self.coordinates_names):
                    if param.grad is None:
                        continue
                    grad = param.grad.data
                    grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)*group['learning_rate']
                    param.data -= grad
                    param_data = torch.div(param.data, torch.pi, rounding_mode='trunc')
                    param.data-=param_data*torch.pi                
    
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
                    

class Axon_layer(nn.Module):
    
    def Ordered_coordinates(self):
        for d in range(self.synapses_weight.size()[0]):
            print(f"переопределяю координаты в синаптическом слое {d}")
            size = np.shape(self.synapses_weight[d].detach().numpy())
            num_dim = self.synapses_weight[d].dim()
            for index in np.ndindex(size):
                for coordinate_layer_num in range(len(self.synapses_coordinates)):
                    if coordinate_layer_num<=num_dim-1:
                        rs = self.output_size[coordinate_layer_num]/size[coordinate_layer_num]
                        self.synapses_coordinates[coordinate_layer_num].data[d][index] = torch.round(index[coordinate_layer_num]*rs)
                    
   
    def __init__(self, input_size = (1024,1024), output_size = (1024,1024), num_sinapse=10, window_width=1, bs = 4,
                 stop_weight = False):
        super(Axon_layer, self).__init__()
        
        self.ww = window_width
        self.smoothing = nn.Tanh()
        synapses_weight = torch.stack([torch.zeros(input_size)] * num_sinapse)
        

        synapses_coordinates = ()
        for j in output_size:
            synapses_axis_coordinates = torch.rand(synapses_weight.size())*torch.pi #j
            synapses_coordinates+=(synapses_axis_coordinates,)
            
        if stop_weight == False:
            self.synapses_weight = nn.Parameter(synapses_weight)
        else:
            self.synapses_weight = nn.Parameter(synapses_weight, requires_grad = False)
            
        self.synapses_coordinates = nn.ParameterList(synapses_coordinates)
        self.output_size = nn.Parameter(torch.tensor(output_size), requires_grad = False)
        self.output = torch.zeros(output_size)

        
        self.synaptic_displacement_list = self.give_window(window_width, len(self.output.size()))
        self.reset_parameters()
        
        ensum_string = ""
     
        if str(type(input_size)) != "<class 'int'>":
            for index in range(1+len(input_size)):
                ensum_string += chr(65 + index)
        else:
            for index in [0, 1]:
                ensum_string += chr(65 + index)
        self.ensum_string = f"{ensum_string}, Z{ensum_string[1:]} ->Z{ensum_string}"


    def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.synapses_weight, a=math.sqrt(5))
        nn.init.uniform_(self.synapses_weight, a = -1, b=1)
    
    def to_decard_idx(self, angle_coord, size):
        a = torch.arcsin(torch.sin(angle_coord))
        b = torch.cos(angle_coord)
        c = (1-torch.sin(angle_coord)**2+0.000001)**0.5
        f = a*b/c
        f = 2*f/torch.pi
        f = f+1
        f = f/2
        f = f*(size - 1)
        return f
    
    def activation(self, signal, output):
        
        signal = torch.einsum(self.ensum_string, self.synapses_weight, signal)
        
        
        synapses_coordinates = ()
        
        for coord in self.synapses_coordinates:
            coord = torch.stack([coord] * signal.size()[0])
            synapses_coordinates+=(coord,)
        
        batch_coord = torch.ones(synapses_coordinates[0].size()).long().to(synapses_coordinates[0].device, dtype=torch.long)
        for i in range(signal.size()[0]):
            batch_coord[i]*=i
            
        variability_synapses_coordinates = ()
        for dim, idx in enumerate(synapses_coordinates): 
            list_of_coordinates_variability = []
            for delta in self.synaptic_displacement_list:
                delta_idx = idx+torch.pi*delta[dim]/(self.output_size[dim]-1)
                decard_idx_delta = self.to_decard_idx(delta_idx, self.output_size[dim])
                round_delta_idx = torch.round(decard_idx_delta).to(torch.int64)  
                list_of_coordinates_variability.append(round_delta_idx)
            new_tensor = torch.stack(list_of_coordinates_variability, dim=-1)
            variability_synapses_coordinates = variability_synapses_coordinates+(new_tensor,)
        
        
        batch_coord = torch.stack([batch_coord] * variability_synapses_coordinates[0].size()[-1], dim=-1)
        signal = torch.stack([signal] * variability_synapses_coordinates[0].size()[-1], dim=-1)
        target_idx = ()
        count = 0
        for round_delta_idx, idx in zip(variability_synapses_coordinates, synapses_coordinates):
            idx = torch.stack([idx] * round_delta_idx.size()[-1], dim=-1)
            decard_idx = self.to_decard_idx(idx, self.output_size[dim])
            target_idx+=(round_delta_idx,)
            if count == 0:
                src = (round_delta_idx-decard_idx)**2
            else:
                src+=(round_delta_idx-decard_idx)**2
            count+=1 
        
        src=torch.sqrt(src)
        src = 6*(self.ww - 2*src)/self.ww
        src = 1/(1+torch.exp(-src))
        
        signal = signal*src
        target_idx = (batch_coord,) + target_idx
        
        output = torch.index_put(output, target_idx, signal, accumulate=True)
            
        return output
        
    
    def forward(self, signal):
        
        out_shape = (signal.size()[0],)+self.output.size()
        output = torch.zeros(out_shape).to(signal.device)
        
        output = self.activation(signal, output)
        return output
    
    def give_window(self, window_width, num_dim):
        d = [list(range(-window_width, window_width+1)) for _ in range(num_dim)]
        comb = list(itertools.product(*d))
        return comb

    def extra_repr(self):
        return f'in_features={self.synapses_weight[0].size()}, out_features={self.output.size()}'
    

class Dendrite_layer(nn.Module):
    
    def Ordered_coordinates(self):
        for d in range(self.synapses_weight.size()[0]):
            print(f"переопределяю координаты в синаптическом слое {d}")
            size = np.shape(self.synapses_weight[d].detach().numpy())
            num_dim = self.synapses_weight[d].dim()
            for index in np.ndindex(size):
                for coordinate_layer_num in range(len(self.synapses_coordinates)):
                    if coordinate_layer_num<=num_dim-1:
                        rs = self.output_size[coordinate_layer_num]/size[coordinate_layer_num]
                        self.synapses_coordinates[coordinate_layer_num].data[d][index] = torch.round(index[coordinate_layer_num]*rs)
    
    def reverse_index_put(self, output, input_idx, input_data):
        output = output + input_data[input_idx]
        return output
   
    def __init__(self, input_size = (1024,1024), output_size = (1024,1024), num_sinapse=10, window_width=1, bs = 4,
                 stop_weight = False):
        super(Dendrite_layer, self).__init__()
        
        self.smoothing = nn.Tanh()
        self.ww = window_width
        synapses_weight = torch.stack([torch.zeros(output_size)] * num_sinapse)
        synapses_coordinates = ()
        
        for j in input_size:
            synapses_axis_coordinates = torch.rand(synapses_weight.size())*torch.pi #j
            synapses_coordinates+=(synapses_axis_coordinates,)
            
        if stop_weight == False:
            self.synapses_weight = nn.Parameter(synapses_weight)
        else:
            self.synapses_weight = nn.Parameter(synapses_weight, requires_grad = False)
            
        self.synapses_coordinates = nn.ParameterList(synapses_coordinates)
        self.output_size = nn.Parameter(torch.tensor(output_size), requires_grad = False)
        self.input_size = nn.Parameter(torch.tensor(input_size), requires_grad = False)
        self.output = torch.zeros(output_size)
        
        self.synaptic_displacement_list = self.give_window(window_width, len(self.input_size))
        self.reset_parameters()
        
        ensum_string = ""
     
        if str(type(self.synapses_weight.size())) != "<class 'int'>":
            for index in range(len(self.synapses_weight.size())):
                ensum_string += chr(65 + index)
            self.ensum_string = f"Z{ensum_string}X -> Z{ensum_string[1:]}"
        else:
            self.ensum_string = "ZAX -> Z"            
        

    def reset_parameters(self):
        nn.init.uniform_(self.synapses_weight, a = -1, b=1)
    
    def to_decard_idx(self, angle_coord, size):
        a = torch.arcsin(torch.sin(angle_coord))
        b = torch.cos(angle_coord)
        c = (1-torch.sin(angle_coord)**2+0.000001)**0.5
        f = a*b/c
        f = 2*f/torch.pi
        f = f+1
        f = f/2
        f = f*(size - 1)
        return f
    
    def activation(self, signal):
        synapses_weight = self.synapses_weight
        synapses_coordinates = ()

        for coord in self.synapses_coordinates:
            coord = torch.stack([coord] * signal.size()[0])
            synapses_coordinates+=(coord,)
        
        batch_synapses_weight = torch.stack([synapses_weight] * signal.size()[0])
        batch_coord = torch.ones(synapses_coordinates[0].size()).long().to(synapses_coordinates[0].device, dtype=torch.long)
        
        for i in range(signal.size()[0]):
            batch_coord[i]*=i
  
        output = torch.zeros((signal.size()[0],)+self.synapses_weight.size()).to(signal.device)
        
        variability_synapses_coordinates = ()
        for dim, idx in enumerate(synapses_coordinates): 
            list_of_coordinates_variability = []
            for delta in self.synaptic_displacement_list:
                delta_idx = idx+torch.pi*delta[dim]/(self.input_size[dim] - 1)
                decard_idx_delta = self.to_decard_idx(delta_idx, self.input_size[dim])
                round_delta_idx = torch.round(decard_idx_delta).to(torch.int64)  
                list_of_coordinates_variability.append(round_delta_idx)
            new_tensor = torch.stack(list_of_coordinates_variability, dim=-1)
            variability_synapses_coordinates = variability_synapses_coordinates+(new_tensor,)
        
        output = torch.stack([output] * variability_synapses_coordinates[0].size()[-1], dim=-1)
        batch_coord = torch.stack([batch_coord] * variability_synapses_coordinates[0].size()[-1], dim=-1)
        batch_synapses_weight = torch.stack([batch_synapses_weight] * variability_synapses_coordinates[0].size()[-1], dim=-1)
        target_idx = ()
        count = 0
        for round_delta_idx, idx in zip(variability_synapses_coordinates, synapses_coordinates):
            idx = torch.stack([idx] * round_delta_idx.size()[-1], dim=-1)
            decard_idx = self.to_decard_idx(idx, self.input_size[dim])
            target_idx+=(round_delta_idx,)
            if count == 0:
                src = (round_delta_idx-decard_idx)**2
            else:
                src+=(round_delta_idx-decard_idx)**2
            count+=1 
        
        src=torch.sqrt(src)
        src = 6*(self.ww - 2*src)/self.ww
        src = 1/(1+torch.exp(-src))
        
        target_idx = (batch_coord,) + target_idx
        output += self.reverse_index_put(output, target_idx, signal)*batch_synapses_weight*src
        output = torch.einsum(self.ensum_string, output)
        
        return output
        
    
    def forward(self, signal):               
        out_shape = (signal.size()[0],)+self.output.size()
        output = torch.zeros(out_shape).to(signal.device)
        output = self.activation(signal)
        return output
    
    def give_window(self, window_width, num_dim):
        d = [list(range(-window_width, window_width+1)) for _ in range(num_dim)]
        comb = list(itertools.product(*d))
        return comb

    def extra_repr(self):
        return f'in_features={self.synapses_weight[0].size()}, out_features={self.output.size()}'
       



class Flat_Morphic_module(nn.Module):
               
    def __init__(self,
                 input_size = (1024,1024),
                 output_size = (1024,1024),
                 num_sinapse=10,
                 window_width=1,
                 bs = 4,
                 stop_weight = False):
        
        super(Flat_Morphic_module, self).__init__()
        
        self.DL = Dendrite_layer(input_size = input_size,
                                 output_size = output_size,
                                 num_sinapse = num_sinapse,
                                 window_width = window_width,
                                 bs = bs,
                                 stop_weight = stop_weight) 
        
        self.AL = Axon_layer(input_size = input_size,
                            output_size = output_size,
                            num_sinapse = num_sinapse,
                            window_width = window_width,
                            bs = bs,
                            stop_weight = stop_weight)
        
    def forward(self, x):
        x = x.squeeze(dim=1) 
        y = self.DL(x) + self.AL(x)
        y = y.unsqueeze(1)
        return y
    
class Morphic_module(nn.Module):
               
    def __init__(self,
                 input_size = (1024,1024),
                 output_size = (1024,1024),
                 num_sinapse=10,
                 window_width=1,
                 bs = 4,
                 stop_weight = False):
        
        super(Morphic_module, self).__init__()
        
        self.DL = Dendrite_layer(input_size = input_size,
                                 output_size = output_size,
                                 num_sinapse = num_sinapse,
                                 window_width = window_width,
                                 bs = bs,
                                 stop_weight = stop_weight) 
        
        self.AL = Axon_layer(input_size = input_size,
                            output_size = output_size,
                            num_sinapse = num_sinapse,
                            window_width = window_width,
                            bs = bs,
                            stop_weight = stop_weight)
        
    def forward(self, x):
        x = x.squeeze(dim=1) 
        y = self.DL(x) + self.AL(x)
        y = y.unsqueeze(1)
        return y

class MorphicNetwork(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 num_sinapse,
                 window_width,
                 num_layers,
                 sw):
        
        super(MorphicNetwork, self).__init__()
        
        self.layers = nn.ModuleList([Morphic_module(input_size,
                                                    input_size,
                                                    num_sinapse,
                                                    window_width,
                                                    bs = 4,
                                                    stop_weight = sw) for _ in range(num_layers)])
        self.activation = nn.Tanh()
        self.flatten = nn.Flatten()
        
        num_elements = torch.prod(torch.tensor(input_size))
        
        self.out = nn.Linear(num_elements, num_classes)
  
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.flatten(x)
        x = self.out(x)
        return x

class SmallMorphicNetwork(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 num_sinapse,
                 window_width,
                 num_layers,
                 feauters_size,
                 sw):
        
        super(SmallMorphicNetwork, self).__init__()
        
        track_running_stats = False
        
        self.ww = window_width
        
        self.inp_layers  = Morphic_module(input_size,
                                          feauters_size,
                                          num_sinapse,
                                          window_width,
                                          bs = 4,
                                          stop_weight = sw) 
        
        self.layers = nn.ModuleList([Morphic_module(feauters_size,
                                                    feauters_size,
                                                    num_sinapse,
                                                    window_width,
                                                    bs = 4,
                                                    stop_weight = sw) for _ in range(num_layers)])
        
        self.bns = nn.ModuleList([nn.BatchNorm2d(1, track_running_stats = track_running_stats) for _ in range(num_layers)])
        self.bn1 = nn.BatchNorm2d(1, track_running_stats = track_running_stats)
        self.activation = nn.Tanh()
        self.flatten = nn.Flatten()
        
        num_elements = torch.prod(torch.tensor(feauters_size))
        
        self.out = nn.Linear(num_elements, num_classes)
  
    
    def forward(self, x):
        x = self.inp_layers(x)
        x = self.bn1(x)
        x = self.activation(x)
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
    
    
class FlatMorphicNetwork(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 num_sinapse,
                 window_width,
                 num_layers,
                 num_future,
                 sw):
        
        super(FlatMorphicNetwork, self).__init__()
        
        self.ww = window_width
        
        num_elements = num_future
        inp_elements = torch.prod(torch.tensor(input_size))
        
        self.inp_layers  = Flat_Morphic_module((inp_elements,), 
                                               (num_elements,),
                                               num_sinapse,
                                               window_width,
                                               bs = 4,
                                               stop_weight = sw) 
        
        self.layers = nn.ModuleList([Flat_Morphic_module((num_elements, ),
                                                    (num_elements, ),
                                                    num_sinapse,
                                                    window_width,
                                                    bs = 4,
                                                    stop_weight = sw) for _ in range(num_layers)])
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(1, track_running_stats = False) for _ in range(num_layers)])
        self.bn1 = nn.BatchNorm1d(1, track_running_stats = False)
        self.activation = nn.Tanh()
        self.out = nn.Linear(num_elements, num_classes)
        self.flatten = nn.Flatten()
  
    
    def forward(self, x):
        x = x.squeeze(dim=1) 
        x = self.flatten(x)
        x = self.inp_layers(x)
        x = self.bn1(x)
        x = self.activation(x)
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
        
        x = x.squeeze(dim=1) 
        x = self.out(x)
        return x
 
