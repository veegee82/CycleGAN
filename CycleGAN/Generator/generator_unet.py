import os
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.utils import pad_borders, get_patches


class Generator_UNet_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 activation='relu',
                 normalization='IN',
                 filter_dim=64,
                 loss_name='MSE',
                 use_NN=True,
                 skip=True,
                 depth=5,
                 use_patches=False,
                 patch_size=256,
                 generator='AtoB',
                 scope='Generator_UNet',
                 name='Generator_UNet'):
        super().__init__(scope=scope + '_' + generator, name=name + '_' + generator)

        self.activation = activation
        self.normalization = normalization
        self.filter_dim = filter_dim
        self.use_NN = use_NN
        self.skip = skip
        self.depth = depth
        self.loss_name = loss_name
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.path = os.path.realpath(__file__)


class Generator_UNet_Model(IModel):
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
        self.model_name = self.params.name
        self.activation_down = get_activation(name='lrelu')
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

        def down_block(input, scope, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False):

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
                             stride=1,
                             activation=linear,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             name='conv_2')
                net_conv = net
                net = max_pool(net)
            return net, net_conv

        def up_block(net, net_down, scope, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False):

            with tf.variable_scope(scope):
                for layer in net_down:
                    if net.shape == layer.shape:
                        net = net + layer
                        print("skip")
                net = tf.image.resize_images(net,
                                             size=(int(net.shape[1] * 2),
                                                   int(net.shape[2] * 2)),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



                net = pad_borders(net, k_size, mode="REFLECT")
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

            return net

        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            if self.params.use_patches:
                input = get_patches(input, patch_size=self.params.patch_size)

            net = pad_borders(input, k_size=7, mode="REFLECT")

            net = conv2d(net,
                         self.params.filter_dim,
                         k_size=7,
                         stride=1,
                         activation=self.activation_down,
                         normalization=self.normalization,
                         padding='VALID',
                         is_training=self.is_training,
                         reuse=self.reuse,
                         name='conv_in')

            layer = []

            f_out_max = 0
            for n in range(1,self.params.depth,1):
                net, net_conv = down_block(net,
                                 scope='down_block' + str(n + 1),
                                 f_out=self.params.filter_dim * n,
                                 k_size=3,
                                 activation=self.activation_down,
                                 normalization=self.normalization,
                                 is_training=self.is_training)
                f_out_max = self.params.filter_dim * n
                layer.append(net_conv)

            layer.reverse()

            layer_up = []
            for n in range(0,self.params.depth-1,1):
                net = up_block(net,
                               layer,
                               scope='up_block' + str(n + 1),
                               f_out=f_out_max - (self.params.filter_dim * n),
                               k_size=3,
                               activation=self.activation,
                               normalization=self.normalization,
                               is_training=self.is_training)
                layer_up.append(net)

            net = pad_borders(net, k_size=7, mode="REFLECT")
            net = conv2d(net,
                         f_out=3,
                         k_size=7,
                         stride=1,
                         activation=tf.nn.tanh,
                         is_training=self.is_training,
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
