# encoding: utf-8

import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES']='0' #Set a single gpu
warnings.filterwarnings("ignore")

import tensorflow as tf
import datetime
import numpy as np
import restore 

from argparse import ArgumentParser
from keras.layers import Input,Add, Conv2D, Dense, Lambda, MaxPooling2D, BatchNormalization, Concatenate
from keras.layers import ReLU, Activation, UpSampling2D, PReLU, LeakyReLU, Dropout
from keras.optimizers import SGD, Adam
from keras.activations import sigmoid
from keras.initializers import VarianceScaling, RandomNormal
from keras.models import Model
from keras.regularizers import l1,l2,l1_l2
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras_tqdm import TQDMCallback
from keras import backend as K
from tqdm import tqdm

from tensorflow.keras.utils import OrderedEnqueuer, GeneratorEnqueuer, SequenceEnqueuer

from util import DataLoader, plot_test_images
from losses import psnr3 as psnr
from losses import binary_crossentropy
from losses import VGGLossNoActivation as VGGLoss
from losses import euclidean, charbonnier, cosine,mape,mae,mse,L2Loss,L1Loss,L1L2EucliLoss


l1_reg = l1(1e-3)
l2_reg = l2(1e-3)
l1l2_reg = l1_l2(1e-3)


class RTVSRGAN():
    """ 
    Implementation of RTVSRGAN:

    """
    
    def __init__(self,
                 height_lr=64, width_lr=64, channels=3,
                 upscaling_factor=2, 
                 gen_lr=1e-3, dis_lr=1e-4, ra_lr = 1e-4,
                 loss_weights=[1e-3, 5e-3,1e-2],
                 training_mode=True,
                 colorspace = 'RGB'
    ):
        """
            height_lr: height of the lr image
            width_lr: width of the lr image 
            channels: number of channel of the image
            upscaling_factor= factor upscaling
            lr = learning rate
            training_mode: True or False
            colorspace: 'RGB' or 'YCbCr'
        """
        

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr
        self.training_mode = training_mode

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError(
                'Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.colorspace = colorspace
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.ra_lr = ra_lr


        # Gan setup settings
        self.loss_weights=loss_weights
        self.VGGLoss = VGGLoss(self.shape_hr)
        self.gen_loss =  euclidean #euclidean, charbonnier, mae,mse,L2Loss,L1Loss,L1L2EucliLoss 
        self.content_loss = self.VGGLoss.content_loss 
        self.adversarial_loss = binary_crossentropy
        self.ra_loss = binary_crossentropy

        
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        # If training, build rest of GAN network
        if training_mode:
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.ra_discriminator = self.build_ra_discriminator()
            self.compile_ra_discriminator(self.ra_discriminator)
            self.rtvsrgan = self.build_rtvsrgan()
            self.compile_rtvsrgan(self.rtvsrgan)


    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator_{}X.h5".format(filepath, self.upscaling_factor))
        self.discriminator.save_weights("{}_discriminator_{}X.h5".format(filepath, self.upscaling_factor))


    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        print(">> Loading weights...")
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)


    def build_generator(self):
        varscale = 1.
        
        def SubpixelConv2D(scale=2,name="subpixel"):
            
            def subpixel_shape(input_shape):
                dims = [input_shape[0],
                        None if input_shape[1] is None else input_shape[1] * scale,
                        None if input_shape[2] is None else input_shape[2] * scale,
                        int(input_shape[3] / (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape, name=name)

        inputs = Input(shape=(None, None, self.channels),name='input_1')

        x = Conv2D(filters = 64, kernel_size = (3,3), strides=1, #kernel_regularizer=l2_reg, 
                kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
                padding = "same",name='conv_1')(inputs) #activation='relu', 
        x = PReLU(shared_axes=[1,2])(x)
        x_start = x

        x = Conv2D(filters = 32, kernel_size = (3,3), strides=1, #kernel_regularizer=l2_reg, 
                kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
                padding = "same",name='conv_2')(x) #activation='relu',
        x = PReLU(shared_axes=[1,2])(x)
        #x_start = Lambda(lambda x: x_start * 0.2)(x_start)
        #x = Lambda(lambda x: x * 0.2)(x)
        #x = Add()([x, x_start])
        x = Concatenate()([x, x_start])        
        
        x = Conv2D(filters = self.upscaling_factor**2*self.channels, kernel_size = (3,3), strides=1, #kernel_regularizer=l2_reg, 
                kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
                padding = "same", name='conv_3')(x) 
        
        x = SubpixelConv2D(scale=self.upscaling_factor,name='subpixel_')(x)

        x = Activation('tanh',name='tanh_')(x) 

        model = Model(inputs=inputs, outputs=x)
        model.summary()
        return model
    
    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x) # new
        x = Dense(1)(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x,name='Discriminator')
        #model.summary()
        return model
    

    def build_ra_discriminator(self):
        
        def comput_Ra(x):
            d_output1,d_output2 = x
            real_loss = (d_output1 - K.mean(d_output2))
            fake_loss = (d_output2 - K.mean(d_output1))
            return sigmoid(0.5 * np.add(real_loss, fake_loss))

        # Input Real and Fake images, Dra(Xr, Xf)        
        imgs_hr = Input(shape=self.shape_hr)
        generated_hr = Input(shape=self.shape_hr)

        # C(Xr)
        real = self.discriminator(imgs_hr)
        # C(Xf)
        fake = self.discriminator(generated_hr)

        # Relativistic Discriminator
        Ra_out = Lambda(comput_Ra, name='Ra_out')([real, fake])

        model = Model(inputs=[imgs_hr, generated_hr], outputs=Ra_out,name='ra_discriminator')
        #model.summary()    
        return model

  
    def build_rtvsrgan(self):
        """Create the combined ESRGAN network"""

        # Input LR images
        img_lr = Input(self.shape_lr)
        # Input HR images
        img_hr = Input(self.shape_hr)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)

        generated2_hr = self.generator(img_lr)

        # In the combined model we only train the generator
        self.discriminator.trainable = False
        self.ra_discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.ra_discriminator([img_hr,generated_hr])


        # Create sensible names for outputs in logs
        generated_hr = Lambda(lambda x: x, name='Perceptual')(generated_hr)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)
        generated2_hr = Lambda(lambda x: x, name='Content')(generated2_hr)

        # Create model and compile
        # Using binary_crossentropy with reversed label, to get proper loss, see:
        # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
        model = Model(inputs=[img_lr,img_hr], outputs=[generated_hr,generated_check,generated2_hr],name='GAN')
        #model.summary()  
        return model

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        
        model.compile(
            loss=self.gen_loss,
            optimizer=Adam(lr=self.gen_lr,beta_1=0.9, beta_2=0.999), 
            metrics=[psnr]#,euclidean,charbonnier,L1Loss,L2Loss
        )


    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.adversarial_loss,
            optimizer=Adam(lr=self.dis_lr, beta_1=0.9),
            metrics=['accuracy']
        )
    
    def compile_ra_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.ra_loss,
            optimizer=Adam(lr=self.ra_lr, beta_1=0.9),
            metrics=['accuracy']
        )


    def compile_rtvsrgan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.content_loss,self.adversarial_loss,self.gen_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(lr=self.ra_lr, beta_1=0.9)
        )
    

   
    
    def train_generator(self,
            epochs=None, batch_size=None,
            workers=None,
            max_queue_size=None,
            modelname=None,
            datapath_train=None,
            datapath_validation=None,
            datapath_test=None,
            steps_per_epoch=None,
            steps_per_validation=None,
            crops_per_image=None,
            print_frequency=None,
            log_weight_path=None, 
            log_tensorboard_path=None,
            log_tensorboard_update_freq=None,
            log_test_path=None,
            media_type='i'
        ):

        # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )

        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, 5,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )

        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, modelname),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=60, verbose=1, 
            restore_best_weights=True )
        callbacks.append(earlystopping)

        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, modelname + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True
        )
        callbacks.append(modelcheckpoint)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=50, min_lr=1e-5,verbose=1)
        callbacks.append(reduce_lr)

        # Learning rate scheduler
        def lr_scheduler(epoch, lr):
            factor = 0.5
            decay_step = 100 #100 epochs * 2000 step per epoch = 2x1e5
            if epoch % decay_step == 0 and epoch:
                return lr * factor
            return lr
        lr_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)
        callbacks.append(lr_scheduler)

  
        # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_test_images(
                    self.generator,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=modelname,
                    channels=self.channels,
                    colorspace=self.colorspace))
            callbacks.append(testplotting)

        # Use several workers on CPU for preparing batches
        enqueuer = OrderedEnqueuer(
            train_loader,
            use_multiprocessing=True
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()

        """ enqueuer_val = OrderedEnqueuer(
            validation_loader,
            use_multiprocessing=True
        )
        enqueuer_val.start(workers=workers, max_queue_size=max_queue_size)
        val_generator = enqueuer_val.get() """

        self.generator.fit_generator(
            output_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=False,#workers>1,   
            workers=1 #workers
        )
    

    def train_rtvsrgan(self, 
        epochs=None, batch_size=None, 
        modelname=None, 
        datapath_train=None,
        datapath_validation=None, 
        steps_per_validation=None,
        datapath_test=None, 
        workers=None, max_queue_size=None,
        first_epoch=None,
        print_frequency=None,
        crops_per_image=None,
        log_weight_frequency=None, 
        log_weight_path=None, 
        log_tensorboard_path=None,
        log_tensorboard_update_freq=None,
        log_test_frequency=None,
        log_test_path=None, 
        media_type='i'        
    ):
        """Train the ESRGAN network

        :param int epochs: how many epochs to train the network for
        :param str modelname: name to use for storing model weights etc.
        :param str datapath_train: path for the image files to use for training
        :param str datapath_test: path for the image files to use for testing / plotting
        :param int print_frequency: how often (in epochs) to print progress to terminal. Warning: will run validation inference!
        :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int log_weight_path: where should network weights be saved        
        :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        :param str log_test_path: where should test results be saved
        :param str log_tensorboard_path: where should tensorflow logs be sent
        """

        
        
         # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )

        # Validation data loader
        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )
    
        # Use several workers on CPU for preparing batches
        enqueuer = OrderedEnqueuer(
            train_loader,
            use_multiprocessing=True,
            shuffle=True
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
        
        # Callback: tensorboard
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, modelname),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                update_freq=log_tensorboard_update_freq
            )
            tensorboard.set_model(self.rtvsrgan)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Learning rate scheduler
        def lr_scheduler(epoch, lr):
            factor = 0.5
            decay_step =  [50000,100000,200000,300000]  
            if epoch in decay_step and epoch:
                return lr * factor
            return lr
        lr_scheduler_gan = LearningRateScheduler(lr_scheduler, verbose=1)
        lr_scheduler_gan.set_model(self.rtvsrgan)
        lr_scheduler_gen = LearningRateScheduler(lr_scheduler, verbose=0)
        lr_scheduler_gen.set_model(self.generator)
        lr_scheduler_dis = LearningRateScheduler(lr_scheduler, verbose=0)
        lr_scheduler_dis.set_model(self.discriminator)
        lr_scheduler_ra = LearningRateScheduler(lr_scheduler, verbose=0)
        lr_scheduler_ra.set_model(self.ra_discriminator)

        
        # Callback: format input value
        def named_logs(model, logs):
            """Transform train_on_batch return value to dict expected by on_batch_end callback"""
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        # Shape of output from discriminator
        disciminator_output_shape = list(self.ra_discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape) 
               

        # Each epoch == "update iteration" as defined in the paper        
        print_losses = {"GAN": [], "D": []}
        start_epoch = datetime.datetime.now()
        
        # Random images to go through
        #idxs = np.random.randint(0, len(train_loader), epochs)        
        
        # Loop through epochs / iterations
        for epoch in range(first_epoch, int(epochs)+first_epoch):
            lr_scheduler_gan.on_epoch_begin(epoch)
            lr_scheduler_ra.on_epoch_begin(epoch)
            lr_scheduler_dis.on_epoch_begin(epoch)
            lr_scheduler_gen.on_epoch_begin(epoch)

            # Start epoch time
            if epoch % print_frequency == 0:
                print("\nEpoch {}/{}:".format(epoch+1, epochs+first_epoch))
                start_epoch = datetime.datetime.now()            

            # Train discriminator 
            self.discriminator.trainable = True
            self.ra_discriminator.trainable = True
            
            imgs_lr, imgs_hr = next(output_generator)
            generated_hr = self.generator.predict(imgs_lr)

            real_loss = self.ra_discriminator.train_on_batch([imgs_hr,generated_hr], real)
            #print("Real: ",real_loss)
            fake_loss = self.ra_discriminator.train_on_batch([generated_hr,imgs_hr], fake)
            #print("Fake: ",fake_loss)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

            # Train generator
            self.discriminator.trainable = False
            self.ra_discriminator.trainable = False
            
            for _ in tqdm(range(10),ncols=60,desc=">> Training generator"):
                imgs_lr, imgs_hr = next(output_generator)
                gan_loss = self.rtvsrgan.train_on_batch([imgs_lr,imgs_hr], [imgs_hr,real,imgs_hr])
     
            # Callbacks
            logs = named_logs(self.rtvsrgan, gan_loss)
            tensorboard.on_epoch_end(epoch, logs)
            

            # Save losses            
            print_losses['GAN'].append(gan_loss)
            print_losses['D'].append(discriminator_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['GAN']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print(">> Time: {}s\n>> GAN: {}\n>> Discriminator: {}".format(
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.rtvsrgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_avg_loss)])
                ))
                print_losses = {"GAN": [], "D": []}

                # Run validation inference if specified
                if datapath_validation:
                    validation_losses = self.generator.evaluate_generator(
                        validation_loader,
                        steps=steps_per_validation,
                        use_multiprocessing=False,#workers>1,
                        workers=1
                    )
                    print(">> Validation Losses: {}".format(
                        ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.generator.metrics_names, validation_losses)])
                    ))                

            # If test images are supplied, run model on them and save to log_test_path
            if datapath_test and epoch % log_test_frequency == 0:
                plot_test_images(self.generator, test_loader, datapath_test, log_test_path, epoch, modelname,
                channels = self.channels,colorspace=self.colorspace)

            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
                # Save the network weights
                self.save_weights(os.path.join(log_weight_path, modelname))
    
    def predict(self,
            lr_path = None,
            sr_path = None,
            print_frequency = False,
            qp = 8,
            fps = None,
            media_type = None,
            gpu = False
        ):
        """ lr_videopath: path of video in low resoluiton
            sr_videopath: path to output video 
            print_frequency: print frequncy the time per frame and estimated time, if False no print 
            crf: [0,51] QP parameter 0 is the best quality and 51 is the worst one
            fps: framerate if None is use the same framerate of the LR video
            media_type: type of media 'v' to video and 'i' to image
        """
        if(media_type == 'v'):
            time_elapsed = restore.write_srvideo(self.generator,lr_path,sr_path,self.upscaling_factor,print_frequency=print_frequency,crf=qp,fps=fps,gpu=gpu)
        elif(media_type == 'i'):
            time_elapsed = restore.write_sr_images(self.generator, lr_imagepath=lr_path, sr_imagepath=sr_path,scale=self.upscaling_factor)
        else:
            print(">> Media type not defined or not suported!")
            return 0
        return time_elapsed


def parse_args():
    parser = ArgumentParser(description='SR with model RT-VSRGAN')

    parser.add_argument(
        '-m', '--mode',
        type=str, default='train',
        help='Which mode to run.',
        choices=['train', 'test']
    )

    parser.add_argument(
        '-n', '--modelname',
        type=str, default='RT-VSRGAN-v1',
        help='Name of the model.',
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int, default=10,
        help='Number of epochs.',
    )

    parser.add_argument(
        '-pf', '--print_frequency',
        type=int, default=10,
        help='Number of print frequency.',
    )

    parser.add_argument(
        '-s', '--upscaling_factor',
        type=int, default=2,
        help='Number of upscaling factor.',
    )

    
        
    return  parser.parse_args()
    

# Run the RT-VSRGAN network
if __name__ == "__main__":

    args = parse_args()
    print(">> Creating the RT-VSRGAN network")
    rtvsrgan = RTVSRGAN(height_lr=64, width_lr=64,channels=3,upscaling_factor=args.upscaling_factor,
    gen_lr=1e-3, dis_lr=1e-4, ra_lr = 1e-4,
    loss_weights=[1e-3, 5e-3,1e-2],
    colorspace = 'RGB') #YCbCr


    if (args.mode == 'train'):    
        rtvsrgan.train_generator(
            epochs=args.epochs,
            batch_size=128,
            steps_per_epoch=10, #625
            steps_per_validation=10,
            crops_per_image=4,
            print_frequency=args.print_frequency,
            log_tensorboard_update_freq=10,
            workers=4,
            max_queue_size=1,
            modelname=args.modelname,
            media_type='i',
            datapath_train='../../data/DIV2K_train_HR/', 
            datapath_validation='../../data/DIV2K_valid_HR/', 
            datapath_test='../../data/benchmarks/Set5/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
        )
    if (args.mode == 'test'):
        # Instantiate the RTVSRGAN object
        rtvsrgan.load_weights(weights='../model/{}_{}X.h5'.format(args.modelname,args.upscaling_factor))

        datapath = '../../data/videoset/360p/' 
        outpath = '../out/360p_2X/'
        i=1
        for dirpath, _, filenames in os.walk(datapath):
            for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
                if(i<2): 
                    print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
                    t = rtvsrgan.predict(
                            lr_path=os.path.join(dirpath, filename), 
                            sr_path=outpath+filename.split('.')[0]+'.mp4',
                            print_frequency = True,
                            qp=0,
                            media_type='v',
                            gpu=False
                        )
                i+=1
        
        datapath = '../../data/videoset/540p/' 
        outpath = '../out/540p_2X/'
        i=1
        for dirpath, _, filenames in os.walk(datapath):
            for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
                if(i<1):
                    print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
                    t = rtvsrgan.predict(
                            lr_path=os.path.join(dirpath, filename), 
                            sr_path=outpath+filename.split('.')[0]+'.mp4',
                            print_frequency = True,
                            qp=0,
                            media_type='v',
                            gpu=True
                    )
                i+=1


    

    

 

