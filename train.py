#!/usr/bin/python3
# encoding: utf-8


import os
import sys
import subprocess
import json
os.environ['CUDA_VISIBLE_DEVICES']='0' #Set a single gpu
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
sys.path.append('libs/')  
import gc
import numpy as np
import matplotlib.pyplot as plt
# Import backend without the "Using X Backend" message
from argparse import ArgumentParser
from PIL import Image
from libs.rtvsrgan import RTVSRGAN
from util import plot_test_images, DataLoader
from keras import backend as K


# Sample call
"""
# Train 2X RTVSRGAN
python3 train.py -t ../../data/train_large/ -v ../data/val_large/ -te ../data/benchmarks/Set5/  -ltp ./test/ -sc 2 -e 10000 -spe 200 -pf 50 -s quanti -mn _lr1e3_places365

python3 train.py -t ../../data/train_large/ -v ../data/val_large/ -te ../data/benchmarks/Set5/  -ltp ./test/ -sc 2 -e 10000 -spe 200 -pf 50 -s percept_test -mn _lr1e4_places365


# Train the 4X RTVSRGAN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 4 --scaleFrom 2 --stage all

# Train the 8X RTVSRGAN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 8 --scaleFrom 4 --stage all
"""

def parse_args():
    parser = ArgumentParser(description='Training script for RTVSRGAN')

    parser.add_argument(
        '-s', '--stage',
        type=str, default='all',
        help='Which stage of training to run',
        choices=['all', 'quanti', 'percept', 'quanti_test','percept_test']
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int, default=1000000,
        help='Number epochs per train'
    )

    parser.add_argument(
        '-fe', '--first_epoch',
        type=int, default=0,
        help='Number of the first epoch to start in logs train'
    )

    parser.add_argument(
        '-t', '--train',
        type=str, default='../../data/train_large/',
        help='Folder with training images'
    )

    parser.add_argument(
        '-spe', '--steps_per_epoch',
        type=int, default=2000,
        help='Steps per epoch'
    )

    parser.add_argument(
        '-v', '--validation',
        type=str, default='../data/val_large/',
        help='Folder with validation images'
    )

    parser.add_argument(
        '-spv', '--steps_per_validation',
        type=int, default=10,
        help='Steps per validation'
    )

    parser.add_argument(
        '-te', '--test',
        type=str, default='../data/benchmarks/Set5/',
        help='Folder with testing images'
    )

    parser.add_argument(
        '-pf', '--print_frequency',
        type=int, default=10,
        help='Frequency of print test images'
    )
        
    parser.add_argument(
        '-sc', '--scale',
        type=int, default=2,
        help='How much should we upscale images'
    )

    parser.add_argument(
        '-scf', '--scaleFrom',
        type=int, default=None,
        help='Perform transfer learning from lower-upscale model'
    )
        
    parser.add_argument(
        '-w', '--workers',
        type=int, default=4,
        help='How many workers to user for pre-processing'
    )

    parser.add_argument(
        '-mqs', '--max_queue_size',
        type=int, default=100,
        help='Max queue size to workers'
    )
        
    parser.add_argument(
        '-bs', '--batch_size',
        type=int, default=16,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-cpi', '--crops_per_image',
        type=int, default=4,
        help='Increase in order to reduce random reads on disk (in case of slower SDDs or HDDs)'
    )
        
    parser.add_argument(
        '-wp', '--weight_path',
        type=str, default='./model/',
        help='Where to output weights during training'
    )

    parser.add_argument(
        '-lwf', '--log_weight_frequency',
        type=int, default=1,
        help='Where to output weights during training'
    )

    parser.add_argument(
        '-ltf', '--log_test_frequency',
        type=int, default=30,
        help='Frequency to output test'
    )

    parser.add_argument(
        '-ltuf', '--log_tensorboard_update_freq',
        type=int, default=1,
        help='Frequency of update tensorboard weight'
    )
        
    parser.add_argument(
        '-lp', '--log_path',
        type=str, default='./logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '-ltp', '--log_test_path',
        type=str, default='./test/',
        help='Path to generate images in train'
    )

    parser.add_argument(
        '-hlr', '--height_lr',
        type=int, default=64,
        help='height of lr crop'
    )

    parser.add_argument(
        '-wlr', '--width_lr',
        type=int, default=64,
        help='width of lr crop'
    )

    parser.add_argument(
        '-c', '--channels',
        type=int, default=3,
        help='channels of images'
    )

    parser.add_argument(
        '-cs', '--colorspace',
        type=str, default='RGB',
        help='Colorspace of images, e.g., RGB or YYCbCr'
    )

    parser.add_argument(
        '-mt', '--media_type',
        type=str, default='i',
        help='Type of media i to image or v to video'
    )

    parser.add_argument(
        '-mn', '--modelname',
        type=str, default='_places365',
        help='Name for the model'
    )

         
    return  parser.parse_args()

def reset_layer_names(args):
    '''In case of transfer learning, it's important that the names of the weights match
    between the different networks (e.g. 2X and 4X). This function loads the lower-lever
    SR network from a reset keras session (thus forcing names to start from naming index 0),
    loads the weights onto that network, and saves the weights again with proper names'''

    # Find lower-upscaling model results
    BASE_G = os.path.join(args.weight_path, 'SRResNet'+args.modelname+'_'+str(args.scaleFrom)+'X.h5')
    #BASE_G = os.path.join(args.weight_path, 'SRResNet'+args.modelname+'_generator_'+str(args.scaleFrom)+'X.h5')
    #BASE_D = os.path.join(args.weight_path, 'RTVSRGAN'+args.modelname+'_discriminator_'+str(args.scaleFrom)+'X.h5')
    assert os.path.isfile(BASE_G), 'Could not find '+BASE_G
    #assert os.path.isfile(BASE_D), 'Could not find '+BASE_D
    
    # Load previous model with weights, and re-save weights so that name ordering will match new model
    prev_gan = RTVSRGAN(upscaling_factor=args.scaleFrom)
    prev_gan.load_weights(BASE_G) #prev_gan.load_weights(BASE_G, BASE_D)
    prev_gan.save_weights(args.weight_path+'RTVSRGAN{}'.format(args.modelname))
    del prev_gan
    K.reset_uids()
    gc.collect()
    return BASE_G

def model_freeze_layers(args, rtvsrgan):
    '''In case of transfer learning, this function freezes lower-level generator
    layers according to the scaleFrom argument, and recompiles the model so that
    only the top layer is trained'''

    trainable=False
    for layer in rtvsrgan.generator.layers:
        #print(layer.name)
        if layer.name == 'conv_2':
            trainable = True 
        layer.trainable = trainable

    # Compile generator with frozen layers
    rtvsrgan.compile_generator(rtvsrgan.generator)



def train_generator(args, gan, common, epochs=None):
    '''Just a convenience function for training the GAN'''
    gan.train_generator(
        epochs=epochs,
        modelname='SRResNet'+args.modelname,        
        steps_per_epoch=args.steps_per_epoch,                
        **common
    )


def train_gan(args, gan, common, epochs=None):
    '''Just a convenience function for training the GAN'''
    
    gan.train_rtvsrgan(
        epochs=epochs,
        modelname='RTVSRGAN'+args.modelname,    
        log_weight_frequency=args.log_weight_frequency,
        log_test_frequency=args.log_test_frequency,
        first_epoch=args.first_epoch,
        **common
    )

def test_generator(args, model):
    print("TEST ON VIDEO QUANTITATIVE-ORIENTED")

    datapath = '../data/videoset/720p/' 
    outpath = './out/360p_2X/'
    i=1
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            if(i<2): 
                #print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
                t = model.predict(
                        lr_path=os.path.join(dirpath, filename), 
                        sr_path=outpath+filename.split('.')[0]+'.mp4',
                        print_frequency = False,
                        qp=0,
                        media_type='v',
                        gpu=False
                    )
            i+=1

def measure():
    print("MEASURE ON VIDEO QUANTITATIVE-ORIENTED")

    command = '/bin/rm ./out/360p_2X/videoSRC001_1280x720_30_qp_00.yuv'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    command = '/usr/bin/ffmpeg -i ./out/360p_2X/videoSRC001_1280x720_30_qp_00.mp4 ./out/360p_2X/videoSRC001_1280x720_30_qp_00.yuv'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    command = '/usr/bin/python3 ./vmaf/run_vmaf yuv420p 1280 720 ./out/videoSRC001_1280x720_30_qp_00.yuv \
        ./out/360p_2X/videoSRC001_1280x720_30_qp_00.yuv \
        --out-fmt json'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    out = json.loads(out.decode('utf-8'))
    print(json.dumps(out['aggregate'],sort_keys=True,indent=4))


# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
       
    # Common settings for all training stages
    args_train = { 
        "batch_size": args.batch_size, 
        "steps_per_validation": args.steps_per_validation,
        "crops_per_image": args.crops_per_image,
        "print_frequency": args.print_frequency,
        "log_tensorboard_update_freq": args.log_tensorboard_update_freq,
        "workers": args.workers,
        "max_queue_size": args.max_queue_size,
        "datapath_train": args.train,
        "datapath_validation": args.validation,
        "datapath_test": args.test,
        "log_weight_path": args.weight_path, 
        "log_tensorboard_path": args.log_path,        
        "log_test_path": args.log_test_path,        
        "media_type": args.media_type
    }

    # Specific of the model
    args_model = {
        "height_lr": args.height_lr, 
        "width_lr": args.width_lr, 
        "channels": args.channels,
        "upscaling_factor": args.scale, 
        "colorspace": args.colorspace,        
    }

    # Generator weight paths
    srresnet_path = os.path.join(args.weight_path, 'SRResNet{}_{}X.h5'.format(args.modelname,args.scale))
    rtvsrgan_G_path = os.path.join(args.weight_path, 'RTVSRGAN{}_generator_{}X.h5'.format(args.modelname,args.scale))
    rtvsrgan_D_path = os.path.join(args.weight_path, 'RTVSRGAN{}_discriminator_{}X.h5'.format(args.modelname,args.scale))
    # Generator weight paths
    
    ## FIRST STAGE: TRAINING GENERATOR ONLY WITH MSE LOSS
    ######################################################

    # If we are doing transfer learning, only train top layer of the generator
    # And load weights from lower-upscaling model    
    if args.stage in ['all', 'quanti']:
        if args.scaleFrom:
            print("TRANSFERING LEARN")
            # Ensure proper layer names
            BASE_G = reset_layer_names(args)

            # Load the properly named weights onto this model and freeze lower-level layers
            gan = RTVSRGAN(gen_lr=1e-4, **args_model)
                
            gan.load_weights(srresnet_path) #gan.load_weights(BASE_G, BASE_D, by_name=True)
            
            model_freeze_layers(args, gan)
            gan.generator.summary()
            train_generator(args, gan, args_train, epochs=3)

            # Train entire generator for 3 epochs
            gan = RTVSRGAN(gen_lr=1e-4, **args_model)
            gan.generator.summary()
            gan.load_weights(srresnet_path)
            train_generator(args, gan, args_train, epochs=3)
        else: 
            # As in paper - train for 10 epochs
            gan = RTVSRGAN(gen_lr=1e-6, **args_model) #2*1e-4
            last_srresnet_path="./model/SRResNet_lr1e5_places365_2X.h5"
            gan.load_weights(generator_weights=last_srresnet_path)#Teste
            print("TRAINING GENERATOR QUANTITATIVE-ORIENTED") 
            train_generator(args, gan, args_train, epochs=args.epochs)
            gan.load_weights(generator_weights=srresnet_path)    
            test_generator(args, gan)  
            measure()  

    ## SECOND STAGE: TRAINING GAN WITH HIGH LEARNING RATE
    ######################################################

    # Re-initialize & train the GAN - load just created generator weights
    if args.stage in ['all', 'percept']:
        print("TRAINING RTVSRGAN PERCEPTUAL-ORIENTED")
        gan = RTVSRGAN(gen_lr=1e-4, dis_lr=1e-4, ra_lr = 1e-4, loss_weights=[1., 5e-3,1e-2],
            **args_model)
        #srresnet_path="./model/SRResNet_lr1e6_places365_2X.h5"    
        #gan.load_weights(generator_weights=srresnet_path)
        train_gan(args, gan, args_train, epochs= args.epochs//10 if args.epochs == int(4e5) else args.epochs)
    
    if args.stage in ['quanti_test']:
        gan = RTVSRGAN(**args_model)
        gan.load_weights(generator_weights=srresnet_path)
        test_generator(args, gan)
        measure()
    if args.stage in ['percept_test']:
        gan = RTVSRGAN(**args_model)
        gan.load_weights(generator_weights=rtvsrgan_G_path)
        test_generator(args, gan)
        measure()
