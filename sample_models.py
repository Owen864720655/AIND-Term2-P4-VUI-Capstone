from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Dropout)

# model_0
def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    simp_rnn    = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_1
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    simp_rnn    = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_2
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

# model_3
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    simp_rnn_array = []
    bn_rnn_array = []
    for i in range(recur_layers):
        if i is 0:
            simp_rnn_array.append(GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn'+str(i))(input_data))
        else:
            simp_rnn_array.append(GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn'+str(i))(bn_rnn_array[i-1]))
        bn_rnn_array.append(BatchNormalization(name='bn_rnn'+str(i))(simp_rnn_array[i]))
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn_array[recur_layers-1])
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_4
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn   = Bidirectional(LSTM(units, return_sequences=True, name='lstm'), name='bidirectional')(input_data)
    bn_bi_rnn   = BatchNormalization(name='bn_bi_rnn')(bidir_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_bi_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model_5
def cnn_lstm_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent(LSTM) + convolutional network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn    = LSTM(units, return_sequences=True, implementation=2, name='lstm')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model_6
def cnn_dp_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a recurrent + dropout + convolutional network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    dp_conv_1d  = Dropout(0.5)(conv_1d)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(dp_conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model_7
def cnn_rnn_dp_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, dropout_W = 0.5, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model_8
def cnn_dp_rnn_dp_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    dp_conv_1d  = Dropout(0.5)(conv_1d)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(dp_conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, dropout_W = 0.5, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model_9
def cnn_mp_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ CNN + MaxPooling + BatchNormal + RNN(GRU) + TimeDistributed
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    mp_conv_1d  = MaxPooling1D(2)(conv_1d)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(mp_conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: mp_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def mp_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride // 2

# model_10
def cnn_mp_dp_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ CNN + MaxPooling + Dropout + BatchNormal + RNN + BatchNormal + TimeDistributed
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    mp_conv_1d  = MaxPooling1D(2)(conv_1d)
    dp_conv_1d  = Dropout(0.5)(mp_conv_1d)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(dp_conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, dropout_W = 0.5, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: mp_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# model_11
def dil_cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ CNN(Dilated) + BatchNormal + RNN + BatchNormal + TimeDistributed
    """
    input_data  = Input(name='the_input', shape=(None, input_dim))
    conv_1d     = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    bn_cnn      = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn    = GRU(units, activation='relu', return_sequences=True, implementation=2, name='GRU')(bn_cnn)
    bn_rnn      = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense  = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred      = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=10)
    print(model.summary())
    return model

# model_end
def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    return cnn_dp_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units)
