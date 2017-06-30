
"""GAN that learns to generate samples from N(5,1) given N(0,1) noise"""
import numpy as np
# import plotly.graph_objs as go
# import plotly.offline as py
import os
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Input, concatenate, SimpleRNN
from keras.layers import Convolution1D
from keras.models import Model
from keras.optimizers import Adam

from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model, Sequential
from keras.layers import TimeDistributed, Dropout, Activation, Masking, Reshape, Flatten, Lambda, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop, SGD, Adam, Adadelta
import theano.tensor as T
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from EarlyStoppingPerTask import *
from keras.objectives import mse
from keras.activations import relu
from keras.utils.vis_utils import plot_model
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy, binary_accuracy
generator = ''
discriminiator =''
batchsize =''
gan =''


def my_relu(x, alpha=0.0, max_value=1.0):
    return relu(x, alpha=alpha, max_value=max_value)


def cc_coef(y_true, y_pred):
    mu_y_true = T.mean(y_true, axis=-1, keepdims=True)
    mu_y_pred = T.mean(y_pred, axis=-1, keepdims=True)
    var_pred = T.mean((y_pred - mu_y_pred) * (y_pred - mu_y_pred), axis=-1)
    var_true = T.mean((y_true - mu_y_true) * (y_true - mu_y_true), axis=-1)
    return 1 - 2 * T.mean((y_true - mu_y_true) * (y_pred - mu_y_pred), axis=-1) / (
        var_pred + var_true + T.mean(T.square(mu_y_pred - mu_y_true), axis=-1))


def cc_coef_mean(y_true, y_pred):
    mu_y_true = T.mean(y_true, axis=-1, keepdims=True)
    mu_y_pred = T.mean(y_pred, axis=-1, keepdims=True)
    cc = 1 - T.mean(2 * T.mean((y_true - mu_y_true) * (y_pred - mu_y_pred), axis=-1) / (
    T.var(y_true, axis=-1) + T.var(y_pred, axis=-1) + T.mean(T.square(mu_y_pred - mu_y_true), axis=-1)))
    return T.switch(T.isnan(cc), 0, cc)


def build_model_irnn_face(max_frames_x, input_dim, output_dim, nb_units, test_cond=0):
    if test_cond == 1:
        input_seqs = Input(shape=(None, input_dim), dtype='float32', name='input_s')
        input_seqs_rnd = Input(shape=(None, 1), dtype='float32', name='input_s_rnd')
    else:
        input_seqs = Input(shape=(max_frames_x, input_dim), dtype='float32', name='input_s')
        input_seqs_rnd = Input(shape=(max_frames_x, 1), dtype='float32', name='input_s_rnd')

    input_seqs_merged = concatenate([input_seqs, input_seqs_rnd])
    # msk_inp = Masking(mask_value=0.0)(input_seqs)
    layer_id = 0
    outs = {}
    outs['layer' + str(layer_id) + '_out'] = Bidirectional(SimpleRNN(nb_units[layer_id], activation='relu',
                                                       use_bias=True, recurrent_initializer='orthogonal',
                                                       bias_initializer='zeros',
                                                       dropout=0.2, recurrent_dropout=0.2,
                                                       kernel_regularizer=None, recurrent_regularizer=None,
                                                       bias_regularizer=None, activity_regularizer=None,
                                                       kernel_constraint=None, recurrent_constraint=None,
                                                       bias_constraint=None, return_sequences=True))(input_seqs_merged)
    layer_id += 1
    while layer_id < len(nb_units):
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(SimpleRNN(nb_units[layer_id], activation='relu',
                                                           use_bias=True, recurrent_initializer='orthogonal',
                                                           bias_initializer='zeros', dropout=0.2,
                                                           recurrent_dropout=0.2, kernel_regularizer=None,
                                                           recurrent_regularizer=None, bias_regularizer=None,
                                                           activity_regularizer=None, kernel_constraint=None,
                                                           recurrent_constraint=None, bias_constraint=None,
                                                           return_sequences=True))\
            (outs['layer' + str(layer_id - 1) + '_out'])
        layer_id += 1
    outs_drp = Dropout(rate=0.2)(outs['layer' + str(layer_id - 1) + '_out'])
    output = TimeDistributed(Dense(output_dim * 1,),
                             batch_input_shape=(max_frames_x, nb_units[-1]),
                             name='output')(outs_drp)
    model = Model(inputs=[input_seqs, input_seqs_rnd], outputs=[output])
    return model


def build_model_blstm_face(max_frames_x, input_dim, output_dim, nb_units, noise_dim, test_cond=0):
    if test_cond == 1:
        input_seqs = Input(shape=(None, input_dim), dtype='float32', name='input_s')
        input_seqs_rnd = Input(shape=(None, noise_dim), dtype='float32', name='input_s_rnd')
    else:
        input_seqs = Input(shape=(max_frames_x, input_dim), dtype='float32', name='input_s')
        input_seqs_rnd = Input(shape=(max_frames_x, noise_dim), dtype='float32', name='input_s_rnd')

    input_seqs_merged = concatenate([input_seqs, input_seqs_rnd])
    # msk_inp = Masking(mask_value=0.0)(input_seqs)
    layer_id = 0
    outs = {}
    if input_dim < 3:
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                                    dropout=0.0, recurrent_dropout=0.2,
                                                                    return_sequences=True))(input_seqs_merged)
    else:
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                                    dropout=0.2, recurrent_dropout=0.2,
                                                                    return_sequences=True))(input_seqs_merged)
    layer_id += 1
    while layer_id < len(nb_units):
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                       dropout=0.2, recurrent_dropout=0.2,
                                                       return_sequences=True))\
            (outs['layer' + str(layer_id - 1) + '_out'])
        layer_id += 1
    outs_drp = Dropout(rate=0.2)(outs['layer' + str(layer_id - 1) + '_out'])
    output = TimeDistributed(Dense(output_dim * 1,),
                             batch_input_shape=(max_frames_x, nb_units[-1]),
                             name='output')(outs_drp)
    model = Model(inputs=[input_seqs, input_seqs_rnd], outputs=[output])
    return model


def model_discrimnator(max_frames_x, input_dim, output_dim, output_dim_dis, nb_units):
    input_seqs1 = Input(shape=(max_frames_x, input_dim), dtype='float32', name='input_s1')
    input_seqs2 = Input(shape=(max_frames_x, output_dim), dtype='float32', name='input_s2')

    input_seqs = concatenate([input_seqs1, input_seqs2])
    layer_id = 0
    outs = {}
    rs = True
    outs['layer' + str(layer_id) + '_out'] = Bidirectional(SimpleRNN(nb_units[layer_id], activation='relu',
                                                       use_bias=True, recurrent_initializer='orthogonal',
                                                       bias_initializer='zeros',
                                                       dropout=0.2, recurrent_dropout=0.2,
                                                       kernel_regularizer=None, recurrent_regularizer=None,
                                                       bias_regularizer=None, activity_regularizer=None,
                                                       kernel_constraint=None, recurrent_constraint=None,
                                                       bias_constraint=None, return_sequences=rs))(input_seqs)
    layer_id += 1
    while layer_id < len(nb_units):
        if len(nb_units) - layer_id == 1:
            rs = False
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(SimpleRNN(nb_units[layer_id], activation='relu',
                                                           use_bias=True, recurrent_initializer='orthogonal',
                                                           bias_initializer='zeros', dropout=0.2,
                                                           recurrent_dropout=0.2, kernel_regularizer=None,
                                                           recurrent_regularizer=None, bias_regularizer=None,
                                                           activity_regularizer=None, kernel_constraint=None,
                                                           recurrent_constraint=None, bias_constraint=None,
                                                           return_sequences=rs))\
            (outs['layer' + str(layer_id - 1) + '_out'])
        layer_id += 1
    outs_drp = Dropout(rate=0.2)(outs['layer' + str(layer_id - 1) + '_out'])
    # outs_drp_flat = Flatten()(outs_drp)
    output = Dense(output_dim_dis, activation='softmax', name='output-dis')(outs_drp)
    model = Model(inputs=[input_seqs1, input_seqs2], outputs=[output])
    return model


def model_discrimnator_blstm(max_frames_x, input_dim, output_dim, output_dim_dis, nb_units):
    input_seqs1 = Input(shape=(max_frames_x, input_dim), dtype='float32', name='input_s1')
    input_seqs2 = Input(shape=(max_frames_x, output_dim), dtype='float32', name='input_s2')

    input_seqs = concatenate([input_seqs1, input_seqs2])
    layer_id = 0
    outs = {}
    rs = True
    if len(nb_units) - layer_id == 1:
        rs = False
    if input_dim < 3:
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                               dropout=0.0, recurrent_dropout=0.2,
                                                               return_sequences=rs))(input_seqs)
    else:
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                                    dropout=0.2, recurrent_dropout=0.2,
                                                                    return_sequences=rs))(input_seqs)
    layer_id += 1
    while layer_id < len(nb_units):
        if len(nb_units) - layer_id == 1:
            rs = False
        outs['layer' + str(layer_id) + '_out'] = Bidirectional(LSTM(nb_units[layer_id],
                                                               dropout=0.2, recurrent_dropout=0.2,
                                                               return_sequences=rs))\
            (outs['layer' + str(layer_id - 1) + '_out'])
        layer_id += 1
    outs_drp = Dropout(rate=0.2)(outs['layer' + str(layer_id - 1) + '_out'])
    # outs_drp_flat = Flatten()(outs_drp)
    output = Dense(output_dim_dis, activation='softmax', name='output-dis')(outs_drp)
    model = Model(inputs=[input_seqs1, input_seqs2], outputs=[output])
    return model


def accuracy_sup(y_true, y_pred):

    return T.mean(categorical_accuracy(y_true[(1-y_true[:, 0]).nonzero(), 1:], y_pred[(1-y_true[:, 0]).nonzero(), 1:]))


def accuracy_unsup(y_true, y_pred):
    return binary_accuracy(y_true[:, 0], y_pred[:, 0])


def model_generator(max_frames_x, input_dim, output_dim, nb_units):
    return build_model_irnn_face(max_frames_x, input_dim, output_dim, nb_units)


def semi_sup_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * T.makeKeepDims(y_true, 1-y_true[:, 0], axis=-1) + \
           categorical_crossentropy(T.stack([y_true[:, 0], T.sum(y_true[:, 1:], axis=1, keepdims=False)], axis=1),
                                    T.stack([y_pred[:, 0], T.sum(y_pred[:, 1:], axis=1, keepdims=False)], axis=1))

# loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_unl))) \
#                               + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_fake)))

def build_model_gan(max_frames_x, input_dim, output_dim, output_dim_dis, nb_units, learning_r, optimizer, loss, optimizer_dis, network_cond, noise_dim):
    # global generator, discriminator, batch_size, gan
    if network_cond == 'IRNN':
        generator = build_model_irnn_face(max_frames_x, input_dim, output_dim, nb_units)
    elif network_cond == 'BLSTM':
        generator = build_model_blstm_face(max_frames_x, input_dim, output_dim, nb_units, noise_dim)
    if optimizer == 'SGD':
        optimizer_fun = SGD(lr=learning_r, decay=learning_r*0.1, momentum=0.95)
    elif optimizer == 'RMSprop':
        optimizer_fun = RMSprop(lr=learning_r, rho=0.9, epsilon=1e-08)
    elif optimizer == 'Adam':
        optimizer_fun = Adam(lr=learning_r)
    elif optimizer == 'Adadelta':
        optimizer_fun = Adadelta(lr=learning_r)
    generator.compile(loss='mse', optimizer=optimizer_fun)
    if network_cond == 'IRNN':
        discriminator = model_discrimnator(max_frames_x, input_dim, output_dim, nb_units)
    elif network_cond == 'BLSTM':
        discriminator = model_discrimnator_blstm(max_frames_x, input_dim, output_dim, output_dim_dis, nb_units)
    if optimizer_dis == 'SGD':
        optimizer_fun = SGD(lr=learning_r, decay=learning_r*0.1, momentum=0.95)
    elif optimizer_dis == 'RMSprop':
        optimizer_fun = RMSprop(lr=learning_r, rho=0.9, epsilon=1e-08)
    elif optimizer_dis == 'Adam':
        optimizer_fun = Adam(lr=learning_r)
    elif optimizer_dis == 'Adadelta':
        optimizer_fun = Adadelta(lr=learning_r)
    from keras.losses import categorical_crossentropy, binary_crossentropy

    discriminator.compile(loss=semi_sup_loss, metrics=[accuracy_sup, accuracy_unsup, 'accuracy'], optimizer=optimizer_fun)

    discriminator.trainable = False  # Freeze discriminator weights
    gan_inputs = generator.inputs
    y = generator(gan_inputs)
    gan_output = discriminator([gan_inputs[0], y])
    gan = Model(inputs=gan_inputs, outputs=gan_output)
    gan.compile(loss=semi_sup_loss, metrics=[accuracy_sup, accuracy_unsup, 'accuracy'], optimizer=optimizer_fun)
    generator.compile(loss=cc_coef, metrics=['mse'], optimizer=optimizer_fun)
    return gan, discriminator, generator



