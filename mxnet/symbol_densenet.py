'''
References:

Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"
'''
#import find_mxnet
#assert find_mxnet
import mxnet as mx


def add_layer(
    x,
    num_channel,
    name,
    pad=1,
    kernel_size=3,
    dropout=0.,
    l2_reg=1e-4):

    x = mx.symbol.BatchNorm(x, eps = l2_reg, name = name + '_batch_norm')
    x = mx.symbol.Activation(data = x, act_type='relu', name = name + '_relu')
    x = mx.symbol.Convolution(
        name=name + '_conv',
        data=x,
        num_filter=num_channel,
        kernel=(kernel_size, kernel_size),
        stride=(1, 1),
        pad=(pad, pad),
        no_bias=True
    )
    if dropout > 0:
        x = mx.symbol.Dropout(x, p = dropout, name = name + '_dropout')
    return x

def dense_block(
    x,
    num_layers,
    growth_rate,
    name,
    dropout=0.,
    l2_reg=1e-4):

    for i in range(num_layers):
        out = add_layer(x, growth_rate, name=name + '_layer_'+str(i), dropout=dropout, l2_reg=l2_reg)
        x = mx.symbol.Concat(x, out, name=name+'_concat_'+str(i))

    return x

def transition_block(
    x,
    num_channel,
    name,
    dropout=0.,
    l2_reg=1e-4):

    x = add_layer(x, num_channel, name = name, pad=0, kernel_size=1, dropout=dropout, l2_reg=l2_reg)
    x = mx.symbol.Pooling(x, name = name + '_pool', global_pool = False, kernel = (2,2), stride = (2,2), pool_type = 'avg')
    return x

def get_symbol(
    num_class,
    num_block,
    num_layer,
    growth_rate,
    dropout=0.,
    l2_reg=1e-4,
    init_channels=16
):
    n_channels = init_channels

    data = mx.symbol.Variable(name='data')
    #label = mx.sym.Variable("label")
    conv = mx.symbol.Convolution(
        name="conv0",
        data=data,
        num_filter=init_channels,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        no_bias=True
    )

    #conv = mx.symbol.Pooling(conv, name='conv0_pool', global_pool=False, kernel=(3,3), stride=(2,2), pool_type ='max')

    for i in range(num_block - 1):
        conv = dense_block(conv, num_layer, growth_rate, name = 'dense'+str(i)+'_',
                        dropout=dropout, l2_reg=l2_reg)
        n_channels += num_layer * growth_rate
        conv = transition_block(conv, n_channels, name = 'trans'+str(i)+'_', dropout=dropout, l2_reg=l2_reg)

    conv = dense_block(conv, num_layer, growth_rate, name = 'last_', dropout=dropout, l2_reg=l2_reg)
    conv = mx.symbol.BatchNorm(conv, eps = l2_reg, name = 'batch_norm_last')
    conv = mx.symbol.Activation(data = conv, act_type='relu', name='relu_last')
    conv = mx.symbol.Pooling(conv, global_pool=True, kernel=(8, 8), pool_type='avg', name = 'global_avg_pool')
    flat = mx.symbol.Flatten(data=conv)
    fc = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc')

    return mx.symbol.SoftmaxOutput(data=fc, name='softmax')
