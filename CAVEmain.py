# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:59:54 2018
Main Code
@author: XieQi
"""
#import h5py
import os
import skimage.measure
import numpy as np
import scipy.io as sio    
import re
import CAVE_dataReader as Crd
import tensorflow as tf
import MyLib as ML
import random 
import MHFnet as MHFnet
import argparse
from tensorflow.keras.layers import Layer
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# FLAGS参数设置
def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Testing Configuration")

    # Mode: train, test, testAll for test all sample
    parser.add_argument('--mode', type=str, default='train', 
                        help='Mode: train or test or testAll.')

    # Prepare Data: if reprepare data samples for training and testing
    parser.add_argument('--Prepare', type=str, default='No', 
                        help='Prepare data: Yes or No.')

    # Output channel number
    parser.add_argument('--outDim', type=int, default=31, 
                        help='Output channel number.')

    # The rank of Y_hat
    parser.add_argument('--upRank', type=int, default=12, 
                        help='UpRank number.')

    # Alpha
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='Alpha parameter (lambda).')

    # Beta
    parser.add_argument('--beta', type=float, default=0.01, 
                        help='Beta parameter (lambda).')

    # The stage number (HSInet layers)
    parser.add_argument('--HSInetL', type=int, default=20, 
                        help='Layer number of HSInet.')

    # The level number of the resnet for the proximal operator
    parser.add_argument('--subnetL', type=int, default=2, 
                        help='Layer number of subnet.')

    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Learning rate.')

    # Epoch number
    parser.add_argument('--epoch', type=int, default=5, 
                        help='Number of epochs.')

    # Path of testing sample
    parser.add_argument('--test_data_name', type=str, default='TestSample',
                        help='File pattern for eval data.')

    # Path of training result
    parser.add_argument('--train_dir', type=str, default='temp/TrainedNet/',
                        help='Directory to keep training outputs.')

    # Path of the testing result
    parser.add_argument('--test_dir', type=str, default='TestResult/Result/',
                        help='Directory to keep evaluation outputs.')

    # The size of training samples
    parser.add_argument('--image_size', type=int, default=96, 
                        help='Image side length.')

    # The iteration number in each epoch
    parser.add_argument('--BatchIter', type=int, default=20, 
                        help='Number of training iterations in each epoch.')

    # The batch size
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size.')

    # Number of GPUs used for training
    parser.add_argument('--num_gpus', type=int, default=0, 
                        help='Number of GPUs used for training (0 or 1).')

    args = parser.parse_args()
    return args

# Example usage:

#==============================================================================#
#test
class CustomLossLayer(Layer):
    def __init__(self, alpha, beta, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, X, outX, YA, E):
        # Perform TensorFlow operations inside the call function
        mse_loss = tf.reduce_mean(tf.square(X - outX))
        supervision_loss = self.alpha * tf.reduce_mean(tf.square(X - YA))
        error_loss = self.beta * tf.reduce_mean(tf.square(E))
        total_loss = mse_loss + supervision_loss + error_loss
        return total_loss

class IterativeLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        super(IterativeLossLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, X, ListX, HSInetL):
        total_loss = 0
        for i in range(HSInetL - 1):
            total_loss += self.alpha * tf.reduce_mean(tf.square(X - ListX[i]))
        return total_loss


# def test():
#     data = sio.loadmat(FLAGS.test_data_name)
#     Y    = data['RGB']
#     Z    = data['Zmsi']
#     X    = data['msi']   
        
#     ## banchsize H W C
#     inY = np.expand_dims(Y, axis = 0)
#     inY = tf.to_float(inY)
    
#     inZ = np.expand_dims(Z, axis = 0)
#     inZ = tf.to_float(inZ)
    
#     inX = np.expand_dims(X, axis = 0)
#     inX = tf.to_float(inX)
    
#     iniA     = 0
#     iniUp3x3 = 0
    
#     outX, X1, YA, _, HY = MHFnet.HSInet(inY,inZ, iniUp3x3, iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    

#     config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
#     config.gpu_options.allow_growth = True
#     saver = tf.train.Saver(max_to_keep = 5)
#     save_path = FLAGS.train_dir
    
#     with tf.Session(config=config) as sess:        
#        ckpt = tf.train.latest_checkpoint(save_path)
#        saver.restore(sess, ckpt) 
#        pred_X,pred_YA,pred_HY,inX = sess.run([outX, YA, HY, inX])     
    
#     toshow  = np.hstack((ML.normalized(ML.get3band_of_tensor(pred_HY)),ML.get3band_of_tensor(pred_YA)))
#     toshow2 = np.hstack((ML.get3band_of_tensor(pred_X),ML.get3band_of_tensor(inX)))
#     toshow  = np.vstack((toshow,toshow2))
#     print('The vasaul result of Y_hat (left upper), Y*A (right upper), fusion result (left lower) and ground truth (right lower)')
#     ML.imwrite(toshow)
#     ML.imshow(toshow)

# import tensorflow as tf
# import scipy.io as sio
# import numpy as np
# from tensorflow.keras.models import load_model  # Used for loading saved models in TF2.x
from keras.layers import TFSMLayer  # Import TFSMLayer for TensorFlow SavedModel

def test():
    data = sio.loadmat(FLAGS.test_data_name)
    Y = data['RGB']
    Z = data['Zmsi']
    X = data['msi']
    
    # Batch size H W C
    inY = np.expand_dims(Y, axis=0)
    inY = tf.cast(inY, tf.float32)
    
    inZ = np.expand_dims(Z, axis=0)
    inZ = tf.cast(inZ, tf.float32)
    
    inX = np.expand_dims(X, axis=0)
    inX = tf.cast(inX, tf.float32)
    
    iniA = 0
    iniUp3x3 = 0
    
    # Replace MHFnet.HSInet with the appropriate method to call the model
    outX, X1, YA, _, HY = MHFnet.HSInet(inY, inZ, iniUp3x3, iniA, FLAGS.upRank, FLAGS.outDim, FLAGS.HSInetL, FLAGS.subnetL)
    
    # No more session management in TF2.x, models are executed eagerly
    # Load the saved model using `tf.saved_model` or `tf.keras.models.load_model`
    checkpoint = tf.train.Checkpoint(MHFnet=MHFnet)  # Assuming `MHFnet` contains the model or layers you are using
    checkpoint_dir = FLAGS.train_dir  # Directory where checkpoints are saved

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()  # Restore checkpoint weights
        print(f"Restored checkpoint from {latest_checkpoint}")
    else:
        print("No checkpoint found. Please make sure you have saved checkpoints in the correct directory.")
    
    # Start a TensorFlow session to run the model (no need to create a new session in TF2, it's handled automatically)
    pred_X, pred_YA, pred_HY = outX, YA, HY  # Assuming these are the outputs you want to use

    # Visualize the results
    toshow = np.hstack((ML.normalized(ML.get3band_of_tensor(pred_HY)), ML.get3band_of_tensor(pred_YA)))
    toshow2 = np.hstack((ML.get3band_of_tensor(pred_X), ML.get3band_of_tensor(inX)))
    toshow = np.vstack((toshow, toshow2))
    
    print('The visual result of Y_hat (left upper), Y*A (right upper), fusion result (left lower) and ground truth (right lower)')
    ML.imwrite(toshow)
    ML.imshow(toshow)


#==============================================================================#
#train
# def train():
#    data = sio.loadmat(FLAGS.Rdir)
#    R    = data['A']
#     data = sio.loadmat('rowData/CAVEdata/response coefficient')
#     R    = data['R']
#     C    = data['C']
#     Crd.PrepareDataAndiniValue(R,C,FLAGS.Prepare)    
#     random.seed( 1 )  

#     ## 变为4D张量 banchsize H W C
#     iniData1 = sio.loadmat("CAVEdata/iniA")
#     iniA         = iniData1['iniA'] 
#     iniData2= sio.loadmat("CAVEdata/iniUp")
#     iniUp3x3 = iniData2['iniUp1']
                
#     # X       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # HrHS (None,96,96,31)
#     # Y       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))  # HrMS (None,96,96,3)
#     # Z       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size/32, FLAGS.image_size/32, FLAGS.outDim)) # LrHS (None,3,3,31)
#     X = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.outDim), name='HrHS')  # HrHS (None, 96, 96, 31)
#     Y = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3), name='HrMS')  # HrMS (None, 96, 96, 3)
#     Z = tf.keras.Input(shape=(FLAGS.image_size // 32, FLAGS.image_size // 32, FLAGS.outDim), name='LrHS')  # LrHS (None, 3, 3, 31)
#     outX, ListX, YA, E, HY  = MHFnet.HSInet(Y, Z, iniUp3x3,iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
#     # loss function
# # Example usage in your training function or model

#     lr_ = FLAGS.learning_rate
#     # lr  = tf.placeholder(tf.float32 ,shape = [])
#     # lr = tf.Variable(initial_value=0.0001, trainable=False, dtype=tf.float32, name='learning_rate')  # Example initial value
#     lr = 0.0001
#     # g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam
#     # g_optim = tf.keras.optimizers.Adam(learning_rate=lr).minimize(loss)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#     with tf.GradientTape() as tape:
#         loss_layer = CustomLossLayer(FLAGS.alpha, FLAGS.beta)

#         # Use the layer in your computation
#         loss_value = loss_layer(X, outX, YA, E)

#         # loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.alpha*tf.reduce_mean(tf.square(X - YA))+ FLAGS.beta*tf.reduce_mean(tf.square(E))  # supervised MSE loss
#         # for i in range(FLAGS.HSInetL-1):
#         #     loss = loss + FLAGS.alpha*tf.reduce_mean(tf.square(X - ListX[i]))
        
#         loss_layer2 = IterativeLossLayer(alpha=FLAGS.alpha)

#         # Calculate the loss in your train function
#         iterative_loss = loss_layer2(X, ListX, HSInetL=FLAGS.HSInetL)
#         loss_value += iterative_loss
#     grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#==============================================================================#
#train
# def train():
#     # Prepare data
#     data = sio.loadmat('rowData/CAVEdata/response coefficient')
#     R = data['R']
#     C = data['C']
#     Crd.PrepareDataAndiniValue(R, C, FLAGS.Prepare)
#     random.seed(1)

#     iniData1 = sio.loadmat("CAVEdata/iniA")
#     iniA = iniData1['iniA']
#     iniData2 = sio.loadmat("CAVEdata/iniUp")
#     iniUp3x3 = iniData2['iniUp1']

#     # Define inputs
#     X_input = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.outDim), name='HrHS')  # HrHS (None, 96, 96, 31)
#     Y_input = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3), name='HrMS')  # HrMS (None, 96, 96, 3)
#     Z_input = tf.keras.Input(shape=(FLAGS.image_size // 32, FLAGS.image_size // 32, FLAGS.outDim), name='LrHS')  # LrHS (None, 3, 3, 31)

#     # Get outputs from MHFnet
#     outX, ListX, YA, E, HY = MHFnet.HSInet(Y_input, Z_input, iniUp3x3, iniA, FLAGS.upRank, FLAGS.outDim, FLAGS.HSInetL, FLAGS.subnetL)

#     # Define model
#     model = tf.keras.Model(inputs=[X_input, Y_input, Z_input], outputs=[outX, ListX, YA, E, HY])

#     # Define optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

#     # Create example input data (you'll need to replace this with actual data loading logic)
#     example_X = tf.random.normal((2, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))
#     example_Y = tf.random.normal((2, FLAGS.image_size, FLAGS.image_size, 3))
#     example_Z = tf.random.normal((2, FLAGS.image_size // 32, FLAGS.image_size // 32, FLAGS.outDim))

#     # Loss and gradients calculation
#     with tf.GradientTape() as tape:
#         # Forward pass through the model (use actual data here instead of example_X, example_Y, example_Z)
#         outX, ListX, YA, E, HY = model([example_X, example_Y, example_Z])

#         # Loss from custom loss layers
#         loss_layer = CustomLossLayer(FLAGS.alpha, FLAGS.beta)
#         loss_value = loss_layer(example_X, outX, YA, E)

#         # Iterative loss
#         loss_layer2 = IterativeLossLayer(alpha=FLAGS.alpha)
#         iterative_loss = loss_layer2(example_X, ListX, HSInetL=FLAGS.HSInetL)
#         loss_value += iterative_loss

#     # Get gradients and apply them to trainable variables
#     grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     # print("Trainable variables:")
#     # for var in model.trainable_variables:
#     #     print(var.name)

    
#     # saver setting
#     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#     manager = tf.train.CheckpointManager(checkpoint, directory=FLAGS.train_dir, max_to_keep=5)

#     if manager.latest_checkpoint:
#         checkpoint.restore(manager.latest_checkpoint)
#         print(f"Restored from {manager.latest_checkpoint}")
#     else:
#         print("Starting training from scratch.")

#     # Prepare training and validation data
#     allX, allY = Crd.all_train_data_in()
#     val_h5_X, val_h5_Y, val_h5_Z = Crd.eval_data_in(C, 20)

#     # Training loop
#     for j in range(FLAGS.epoch):
#         if j+1 > (4*FLAGS.epoch/5):
#             lr_ = FLAGS.learning_rate * 0.1

#         Training_Loss = 0
#         for num in range(FLAGS.BatchIter):
#             batch_X, batch_Y, batch_Z = Crd.train_data_in(allX, allY, C, FLAGS.image_size, FLAGS.batch_size)

#             with tf.GradientTape() as tape:
#                 outX, ListX, YA, E, HY = model([batch_X, batch_Y, batch_Z], training=True)
#                 loss_value = compute_loss(batch_X, outX, YA, E, ListX)

#             grads = tape.gradient(loss_value, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#             Training_Loss += loss_value.numpy()

#             if (num + 1) % 200 == 0:
#                 pred_X, pred_ListX, pred_HY, Pred_YA = model([batch_X, batch_Y, batch_Z], training=False)
#                 psnr = skimage.measure.compare_psnr(batch_X, pred_X)
#                 ssim = skimage.measure.compare_ssim(batch_X, pred_X, multichannel=True)
#                 CurLoss = Training_Loss / (num + 1)

#                 print(f'...Training with the {num+1}-th batch ...')
#                 print(f'{j+1} epoch training, learning rate = {lr_:.8f}, Training_Loss = {CurLoss:.4f}, PSNR = {psnr:.4f}, SSIM = {ssim:.4f}')

#         # Save model checkpoint
#         manager.save()

#         # Validate the model
#         Validation_Loss, pred_val = model([val_h5_X, val_h5_Y, val_h5_Z], training=False)
#         psnr_val = skimage.measure.compare_psnr(val_h5_X, pred_val)
#         ssim_val = skimage.measure.compare_ssim(val_h5_X, pred_val, multichannel=True)

#         print(f'The {j+1} epoch is finished, learning rate = {lr_:.8f}, Training_Loss = {Training_Loss/(num+1):.4f}, '
#               f'Validation_Loss = {Validation_Loss:.4f}, PSNR_Valid = {psnr_val:.4f}, SSIM_Valid = {ssim_val:.4f}')
#         print('=========================================')
#         print('*****************************************')

import tensorflow as tf
import scipy.io as sio
import random
import numpy as np
import skimage.measure
import re

# Define the compute_loss function outside the train function
def compute_loss(X, outX, YA, E, ListX, FLAGS):
    loss_layer = CustomLossLayer(FLAGS.alpha, FLAGS.beta)
    loss_value = loss_layer(X, outX, YA, E)

    # Iterative loss
    loss_layer2 = IterativeLossLayer(alpha=FLAGS.alpha)
    iterative_loss = loss_layer2(X, ListX, HSInetL=FLAGS.HSInetL)
    loss_value += iterative_loss
    return loss_value

def train():
    # Prepare data
    data = sio.loadmat('rowData/CAVEdata/response coefficient')
    R = data['R']
    C = data['C']
    Crd.PrepareDataAndiniValue(R, C, FLAGS.Prepare)
    random.seed(1)

    iniData1 = sio.loadmat("CAVEdata/iniA")
    iniA = iniData1['iniA']
    iniData2 = sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']

    # Define inputs
    X_input = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.outDim), name='HrHS')  # HrHS (None, 96, 96, 31)
    Y_input = tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3), name='HrMS')  # HrMS (None, 96, 96, 3)
    Z_input = tf.keras.Input(shape=(FLAGS.image_size // 32, FLAGS.image_size // 32, FLAGS.outDim), name='LrHS')  # LrHS (None, 3, 3, 31)

    # Get outputs from MHFnet
    outX, ListX, YA, E, HY = MHFnet.HSInet(Y_input, Z_input, iniUp3x3, iniA, FLAGS.upRank, FLAGS.outDim, FLAGS.HSInetL, FLAGS.subnetL)

    # Define model
    model = tf.keras.Model(inputs=[X_input, Y_input, Z_input], outputs=[outX, ListX, YA, E, HY])
    lr_ = FLAGS.learning_rate
    learning_rate = 0.0001
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=FLAGS.train_dir, max_to_keep=5)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Starting training from scratch.")

    # Prepare training and validation data
    allX, allY = Crd.all_train_data_in()
    val_h5_X, val_h5_Y, val_h5_Z = Crd.eval_data_in(C, 20)

    # Training loop
    for j in range(FLAGS.epoch):
        # if j+1 > (4*FLAGS.epoch/5):
        #     lr_ = FLAGS.learning_rate * 0.1

        Training_Loss = 0
        for num in range(FLAGS.BatchIter):
            print(f"batch iter is {num}")
            batch_X, batch_Y, batch_Z = Crd.train_data_in(allX, allY, C, FLAGS.image_size, FLAGS.batch_size)
            print(f"batches ka shpe {batch_X.shape, batch_Y.shape, batch_Z.shape}")

            with tf.GradientTape() as tape:
                outX, ListX, YA, E, HY = model([batch_X, batch_Y, batch_Z], training=True)
                loss_value = compute_loss(batch_X, outX, YA, E, ListX, FLAGS)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            Training_Loss += loss_value.numpy()
            print("reached here")
            if (num + 1) % 2 == 0:
                # outX, ListX, YA, E, HY
                pred_X, pred_ListX, pred_YA, pred_E, Pred_HY = model([batch_X, batch_Y, batch_Z], training=False)
                batch_X_np = batch_X.numpy() if isinstance(batch_X, tf.Tensor) else batch_X
                pred_X_np = pred_X.numpy() if isinstance(pred_X, tf.Tensor) else pred_X

                # from skimage.metrics import peak_signal_noise_ratio
                # psnr = peak_signal_noise_ratio(batch_X_np, pred_X_np)
                # from skimage.metrics import structural_similarity
                # print(f"for ssim {batch_X_np.shape, pred_X_np.shape}")
                ssim_values=[]
                psnr_values = []
                # ssim = structural_similarity(batch_X_np, pred_X_np, channel_axis=-1,win_size =7)
                for i in range(2):
                    psnr = peak_signal_noise_ratio(batch_X_np[i], pred_X_np[i])

                    ssim = structural_similarity(batch_X_np[i], pred_X_np[i], channel_axis=-1, win_size=7, data_range=255)
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

                # ssim = skimage.measure.compare_ssim(batch_X, pred_X, multichannel=True)
                CurLoss = Training_Loss / (num + 1)

                print(f'...Training with the {num+1}-th batch ...')
                print(f'{j+1} epoch training, learning rate = {lr_:.8f}, Training_Loss = {CurLoss:.4f}, PSNR = {psnr:.4f}, SSIM = {ssim_values}')

        # Save model checkpoint
        manager.save()
        print("saved the model")
        # Validate the model
        # Validation_Loss, pred_val = model([val_h5_X, val_h5_Y, val_h5_Z], training=False)
        # psnr_val = skimage.measure.compare_psnr(val_h5_X, pred_val)
        # ssim_val = skimage.measure.compare_ssim(val_h5_X, pred_val, multichannel=True)

        # print(f'The {j+1} epoch is finished, learning rate = {lr_:.8f}, Training_Loss = {Training_Loss/(num + 1):.4f}, '
        #       f'Validation_Loss = {Validation_Loss:.4f}, PSNR_Valid = {psnr_val:.4f}, SSIM_Valid = {ssim_val:.4f}')
        print('=========================================')
        print('*****************************************')
                   
#==============================================================================#

def testAll():
    ## test all the testing samples
    iniA         = 0 
    iniUp3x3     = 0
    # Y       = tf.placeholder(tf.float32, shape=(1, 512, 512, 3))  # supervised data (None,64,64,3)
    # Z       = tf.placeholder(tf.float32, shape=(1, 512/32, 512/32, FLAGS.outDim))
    Y = tf.keras.Input(shape=(512, 512, 3), dtype=tf.float16)  # Supervised data (None, 512, 512, 3)
    Z = tf.keras.Input(shape=(512 // 32, 512 // 32, FLAGS.outDim), dtype=tf.float16)  # (None, 16, 16, FLAGS.outDim)

    outX, X1, YA, _, HY = MHFnet.HSInet(Y, Z, iniUp3x3,iniA,FLAGS.upRank,FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    ML.mkdir(FLAGS.test_dir)
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        for root, dirs, files in os.walk('CAVEdata/X/'):
            for i in range(32):       
                data = sio.loadmat("CAVEdata/Y/"+files[i])
                inY  = data['RGB']
                inY  = np.expand_dims(inY, axis = 0)
                data = sio.loadmat("CAVEdata/Z/"+files[i])
                inZ  = data['Zmsi']
                inZ  = np.expand_dims(inZ, axis = 0)
                pred_X,ListX,pred_HY,pred_YA = sess.run([outX, X1, HY, YA],feed_dict={Y:inY,Z:inZ})  
                pred_Lr = ListX[FLAGS.HSInetL-2]
                sio.savemat(FLAGS.test_dir+files[i], {'outX': pred_X,'outLR': pred_Lr,'outHY': pred_HY, 'outYA':pred_YA})     
                print(files[i] + ' done!')


if __name__ == '__main__':
    FLAGS = parse_args()
    
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    with tf.device(dev):
        if FLAGS.mode == 'test': # simple test
            test()
        elif FLAGS.mode == 'testAll': # test all
            testAll()
        elif FLAGS.mode == 'train': # train
            train()

    
  
