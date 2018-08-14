import os
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.utils import get_patches
from tfcore.utilities.utils import pad_borders


class Generator_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 activation='relu',
                 normalization='IN',
                 filter_dim=64,
                 use_NN=True,
                 use_recursice_block=False,
                 units=3,
                 blocks=3,
                 is_training=False,
                 generator='AtoB',
                 scope='Generator',
                 name='Generator'):
        if use_recursice_block:
            appendix = '_recursiv_B' + str(blocks) + '_U' + str(units)
        else:
            appendix = ''

        super().__init__(scope=scope + appendix + '_' + generator, name=name + appendix + '_' + generator)

        self.activation = activation
        self.normalization = normalization
        self.filter_dim = filter_dim
        self.use_NN = use_NN
        self.path = os.path.realpath(__file__)
        self.use_recursice_block = use_recursice_block
        self.units = units
        self.blocks = blocks
        self.is_training = is_training


class Generator_Model(IModel):
    """
    Example of a simple 3 layer generator model for super-resolution
    """

    def __init__(self, sess, params, global_steps, is_training):
        """
        Init of Example Class

        # Arguments
            sess: Tensorflow-Session
            params: Instance of ExampleModel_Params
            global_steps: Globel steps for optimizer
        """
        super().__init__(sess, params, global_steps)
        self.params.lr_lower_bound = 0.0001
        self.model_name = self.params.name
        self.activation = get_activation(name=self.params.activation)
        self.normalization = get_normalization(self.params.normalization)
        self.max_images = 1
        self.is_training = is_training

    def build_model(self, input, is_train=False, reuse=False):
        """
        Build model and create summary

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """
        self.reuse = reuse
        super().build_model(input, is_train, reuse)

        return self.G

    def model(self, input, is_train=False, reuse=False):
        """
        Create generator model

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """

        def recursive_block(input, f_out, activation, share_weights, scope,
                            use_pre_activation=False, normalization=None, is_training=False, units=3):

            def residual_block(net, f_out, activation, share_weights, w1_in, w2_in,
                               scope, use_pre_activation=False, normalization=None, is_training=False):
                with tf.variable_scope(scope):
                    if not share_weights:
                        w1_in = None
                        w2_in = None
                    net = pad_borders(net, 3, mode="REFLECT")
                    net, w1_out = conv2d(net,
                                         f_out=f_out,
                                         k_size=3,
                                         stride=1,
                                         activation=activation,
                                         use_preactivation=use_pre_activation,
                                         is_training=is_training,
                                         normalization=normalization,
                                         use_bias=False,
                                         get_weights=True,
                                         set_weight=w1_in,
                                         padding='VALID',
                                         name='conv_1')
                    net = pad_borders(net, 3, mode="REFLECT")
                    net, w2_out = conv2d(net,
                                         f_out=f_out,
                                         k_size=3,
                                         stride=1,
                                         use_preactivation=use_pre_activation,
                                         is_training=is_training,
                                         normalization=normalization,
                                         use_bias=False,
                                         get_weights=True,
                                         set_weight=w2_in,
                                         padding='VALID',
                                         name='conv_2')

                    return net, w1_out, w2_out

            with tf.variable_scope(scope):
                net = pad_borders(input, 3, mode="REFLECT")
                net_begin = conv2d(net,
                                   f_out=f_out,
                                   k_size=3,
                                   stride=1,
                                   activation=activation,
                                   normalization=normalization,
                                   use_bias=False,
                                   is_training=is_training,
                                   get_weights=False,
                                   padding='VALID',
                                   name='conv_begin')

                net = net_begin

                w1 = None
                w2 = None
                for n in range(units):
                    net, w1, w2 = residual_block(net,
                                                 f_out,
                                                 activation,
                                                 share_weights,
                                                 use_pre_activation=use_pre_activation,
                                                 normalization=normalization,
                                                 is_training=is_training,
                                                 w1_in=w1,
                                                 w2_in=w2,
                                                 scope=scope + '/ResBlock_' + str(n))
                    net = tf.add(net, net_begin)
                return net

        def res_block(input, scope, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False):

            with tf.variable_scope(scope):
                net = pad_borders(input, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             activation=activation,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             name='conv_1')
                net = pad_borders(net, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             activation=linear,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             name='conv_2')
            return net + input

        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            net = pad_borders(input, k_size=7, mode="REFLECT")
            net = conv2d(net,
                         self.params.filter_dim,
                         k_size=7,
                         stride=1,
                         activation=self.activation,
                         normalization=self.normalization,
                         padding='VALID',
                         is_training=self.is_training,
                         reuse=self.reuse,
                         name='conv_in')

            net = conv2d(net,
                         self.params.filter_dim * 2,
                         k_size=3,
                         stride=2,
                         activation=self.activation,
                         normalization=self.normalization,
                         is_training=self.is_training,
                         reuse=self.reuse,
                         padding='SAME',
                         name='conv_1')

            net = conv2d(net,
                         self.params.filter_dim * 4,
                         k_size=3,
                         stride=2,
                         activation=self.activation,
                         normalization=self.normalization,
                         is_training=self.is_training,
                         reuse=self.reuse,
                         padding='SAME',
                         name='conv_2')
            net_begin = net
            if self.params.use_recursice_block:

                for i in range(self.params.blocks):
                    net = recursive_block(net,
                                          f_out=self.params.filter_dim * 4,
                                          activation=self.activation,
                                          normalization=self.normalization,
                                          is_training=self.is_training,
                                          share_weights=True,
                                          units=self.params.units,
                                          scope='Recursive_Block_' + str(i))
                net = net + net_begin
            else:
                for n in range(9):
                    net = res_block(net,
                                    scope='res_block' + str(n + 1),
                                    f_out=self.params.filter_dim * 4,
                                    k_size=3,
                                    activation=self.activation,
                                    normalization=self.normalization,
                                    is_training=self.is_training)

            if self.params.use_NN:
                net = upscale_nearest_neighbor(net,
                                               f_size=self.params.filter_dim * 4,
                                               resize_factor=2,
                                               is_training=not self.params.is_training)


                net = conv2d(net,
                             f_out=self.params.filter_dim * 2,
                             k_size=3,
                             stride=1,
                             activation=self.activation,
                             normalization=self.normalization,
                             is_training=self.is_training,
                             reuse=self.reuse,
                             padding='SAME',
                             name='conv_NN_1')
            else:
                net = deconv2d(net,
                               f_out=self.params.filter_dim * 2,
                               k_size=3,
                               stride=2,
                               activation=self.activation,
                               normalization=self.normalization,
                               is_training=self.is_training,
                               reuse=self.reuse,
                               padding='SAME',
                               name='deconv_1')

            if self.params.use_NN:
                net = upscale_nearest_neighbor(net,
                                               f_size=self.params.filter_dim * 2,
                                               resize_factor=2,
                                               is_training=not self.params.is_training)

                net = conv2d(net,
                             f_out=self.params.filter_dim,
                             k_size=3,
                             stride=1,
                             activation=self.activation,
                             normalization=self.normalization,
                             is_training=self.is_training,
                             reuse=self.reuse,
                             padding='SAME',
                             name='conv_NN_2')
            else:
                net = deconv2d(net,
                               f_out=self.params.filter_dim,
                               k_size=3,
                               stride=2,
                               activation=self.activation,
                               normalization=self.normalization,
                               is_training=self.is_training,
                               reuse=self.reuse,
                               padding='SAME',
                               name='deconv_2')

            net = pad_borders(net, k_size=7, mode="REFLECT")
            net = conv2d(net,
                         f_out=3,
                         k_size=7,
                         stride=1,
                         activation=tf.nn.tanh,
                         reuse=self.reuse,
                         padding='VALID',
                         name='conv_out')

            if not self.reuse:
                self.G_A = self.G = net
                self.reuse = True
            else:
                self.G_B = net

        print(' [*] CycleGAN-Generator loaded...')
        return net
