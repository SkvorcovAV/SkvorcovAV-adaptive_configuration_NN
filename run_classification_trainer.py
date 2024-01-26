import torch
import os
from core.data_manage.classes.base_classes import MNIST_classes
from core.data_manage.standart_data.MNIST import standart_data as MNIST_standart_data
from core.data_manage.standart_data.sinusite import standart_data as sinusite_standart_data
from core.data_manage.standart_data.SCT import standart_data as SCT_standart_data
from core.data_manage.classes.sinusite_standart_classes import *
from core.data_manage.classes.SCT_classes import *
from core.data_manage.datasets.special import *

from core.trainers.classification import Trainer
from core.losses.classification import Combo_loss
import segmentation_models_pytorch as smp
from core.utils.transforms import standart_segmentation_transform, standart_image_transfoms
from core.utils.seed import set_seed
import torchvision

from core.models.custom_classification_models_pytorch.EfficientNetB7 import SimpleCNN
from core.models.morphic_neural_network import *

set_seed(64)


device = "cuda:0"

fast_learning = True
test_mode = False
Delete_background = True

early_stop = 120
num_epochs = 120

clean_segmentation = False 
use_class_weight = True
use_pixel_weight = False
shuffle = True
mix_test = False

batch_size = 140
resize = (28, 28)
Multiclasses_or_multilabel = "ML"
learning_rate = 0.00001

descriptions = "2DMorphicNetwork_ns_20_ww_2_nl_5_fs_28_28_lw_false_new_coord_syst"
useles_classes = MNIST_classes
class_name = "MNIST"

important_classes = ["7"]
target_metrick = {}
target_metrick["optimal_precession"] = 0.8
target_metrick["optimal_recall"] = 0.8
target_metrick["AUROC"] = 0.8

if class_name.find("SCT")!=-1:
    standart_data = SCT_standart_data
elif class_name.find("sinusite")!=-1:
    standart_data = sinusite_standart_data
elif class_name.find("MNIST")!=-1:
    standart_data = MNIST_standart_data
    
name_of_exp = f"{class_name}_batch_size_{str(batch_size)}_lr_{learning_rate}_{descriptions}"
print(name_of_exp)


my_transfoms = standart_image_transfoms()
my_transfoms = my_transfoms.transformation

data_name_weight = standart_data(resize = resize,
                                 out_classes = useles_classes,
                                 batch_size = batch_size,
                                 recalculate=False,
                                 mix_test = mix_test,
                                 shuffle = shuffle)

train_DataLoader, val_DataLoader, test_DataLoader, weight, pixel_weight, list_of_name = data_name_weight

if fast_learning:
    print("Create fast dataloader (train)")
    train_DataLoader = create_fast_dataloader(DataLoader = train_DataLoader, shuffle = shuffle)
    print("Create fast dataloader (validation)")
    val_DataLoader = create_fast_dataloader(DataLoader = val_DataLoader, shuffle = False)
    
if Delete_background==True:
    num_cl = len(useles_classes)
    if weight!=None:
        weight = weight[:,1:].detach().clone()
    if pixel_weight!=None: 
        pixel_weight = pixel_weight[:,1:].detach().clone()
else:
    num_cl = len(useles_classes)+1

    
# from core.models.custom_classification_models_pytorch.med_vit.MedViT import MedViT_large, MedViT_small
# net = MedViT_large()
# net.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=num_cl, bias=True)

# from core.models.custom_classification_models_pytorch.med_vit.MedViT import MedViT_large, MedViT_small



net = SmallMorphicNetwork(input_size = resize,
                          num_classes = num_cl,
                          num_sinapse = 20,
                          window_width = 2,
                          num_layers = 5,
                          feauters_size  = resize,
                          sw = True)

# net = FlatMorphicNetwork(input_size = resize,
#                          num_classes = num_cl,
#                          num_sinapse = 20,
#                          window_width = 4,
#                          num_layers = 5,
#                          num_future  = 800,
#                          sw = False)

optimizer = morphological_net_optimizer(net)

# net = SimpleCNN()


my_loss = Combo_loss(Mode=Multiclasses_or_multilabel, 
                     Class_Weight = weight,
                     Use_focal_loss=True)

My_Trainer = Trainer(net = net,
                     train_dataloader = train_DataLoader,
                     val_dataloader = val_DataLoader,
                     classes = list_of_name,
                     Mode = Multiclasses_or_multilabel,
                     device = device,
                     loss = my_loss,
                     learning_rate = learning_rate,
                     epochs=num_epochs,
                     exp_name = name_of_exp, 
                     transform = my_transfoms,
                     Delete_background = Delete_background,
                     test_mode=test_mode,
                     early_stop = early_stop,
                     important_classes = important_classes,
                     target_metrick =target_metrick)

My_Trainer.optimizer = optimizer

My_Trainer.start_train_cycle()