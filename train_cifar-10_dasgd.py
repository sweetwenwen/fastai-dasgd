import argparse
import sys
from fastai.script import *
from fastai.vision import *
from fastai.vision.models.xresnet2      import *
from fastai.vision.models.xresnet       import xresnet50,xresnet34,xresnet101,xresnet152
from fastai.vision.models.presnet       import PResNet,presnet18,presnet34,presnet50,presnet101,presnet152
from fastai.vision.models.densenet      import densenet_cifar
from fastai.vision.models.vgg           import VGG11, VGG13, VGG16, VGG19
from fastai.vision.models.googlenet     import googlenet_cifar
from fastai.vision.models.mobilenetv2   import mobilenetv2
from fastai.vision.models.resnext       import ResNeXt29_2x64d,ResNeXt29_4x64d,ResNeXt29_8x64d,ResNeXt29_32x4d
from fastai.vision.models.dpn           import DPN26,DPN92
from fastai.vision.models.shufflenetv2  import ShuffleNetV2_0d5

from fastai.distributed import *
from fastai import metrics
torch.backends.cudnn.benchmark = True

path = Path('/root/fastai/cifar10/')
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    
@call_parse
def main(
        gpu:            Param("GPU to run on", str)=            None,
        arch:           Param("Architecture (xresnet34,xresnet50,xresnet101,xresnet152,densenet,googlenet,vgg16,vgg19,mobilenetv2,resnext29_2x64d)", str)='xresnet50',
        local_batch:    Param("Local batch size", int)=         32,
        tao:            Param("Local step size", int)=          4,
        delay:          Param("Delay step of ASGD", int)=       1,
        epoch:          Param("Number of epochs", int)=         50,
        model_number:   Param("Number of models", int)=         32,
        update_proportion:Param("Local update update_proportion*local_weight", float)=      8.0/32.0,
        mom:            Param("Momentum", float)=               0.9,
        lr_max:         Param("Max learning rate", float)=      0.01,
        wd:             Param("Weight decay", float)=           0.01,
        pct_start:      Param("Pct_start", float)=              0.25,
        final_div:      Param("Final_div", float)=              100.0,
        div_factor:     Param("Div_factor", float)=             100.0
        ):    
    
    if   arch=='xresnet50'      : network = xresnet50()
    elif arch=='xresnet34'      : network = xresnet34()
    elif arch=='xresnet101'     : network = xresnet101()
    elif arch=='xresnet152'     : network = xresnet152()
    elif arch=='densenet'       : network = densenet_cifar()
    elif arch=='vgg16'          : network = VGG16()
    elif arch=='vgg19'          : network = VGG19()
    elif arch=='googlenet'      : network = googlenet_cifar()
    elif arch=='mobilenetv2'    : network = mobilenetv2()
    elif arch=='resnext29_2x64d': network = ResNeXt29_2x64d()
    elif arch=='dpn92'          : network = DPN92()
    elif arch=='presnet50'      : network = presnet50()
    elif arch=='shufflenetv2_0d5'      : network = ShuffleNetV2_0d5()
    else                        : raise Exception(f'unknown arch: {arch}')

    number_gpu = torch.cuda.device_count()
    if (model_number<number_gpu):
        print ("The number of models is smaller than the number of GPUs, please reduce the number of GPUs")
        exit()
    if (model_number%number_gpu != 0):
        print ("THe number of models should be multiple of GPUs")
        exit()
    
    step_batch  = local_batch * model_number * tao
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, val_bs=256, bs=step_batch, num_workers=16).normalize(cifar_stats)
    # data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=step_batch, num_workers=16).normalize(cifar_stats)

    print ("{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}\n{:20s}{}".format("training using fit_asynchronous_average_models_one_cycle","MODELS ", model_number,"LOCAL_BATCH",local_batch,"TAO",tao,"DELAY",delay,"UPDATE_PROPORTION",update_proportion,"LR_MAX",lr_max,"DIV_FACTOR",div_factor,"PCT_START",pct_start,"WD",wd,"NETWORK",arch))
    learn_list = {}
    for i in range(model_number):
        learn_list[str(i)] = Learner(data, network)
        learn_list[str(i)].model.cuda(int(i%number_gpu))
    fit_asynchronous_average_models_one_cycle([learn_list[str(i)] for i in range(model_number)], epoch, lr_max, wd=wd, div_factor=div_factor, final_div = final_div, pct_start=pct_start, metrics=[accuracy], model_number=model_number, tao=tao, delay = delay, update_proportion=  update_proportion, number_gpu = number_gpu)



