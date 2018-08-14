import os
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.utils import get_patches


class Patch_Discriminator_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 activation='lrelu',
                 normalization='IN',
                 f_out=1,
                 gamma=1.0,
                 lambd_k=0.001,
                 is_training=True,
                 loss_name='MAE',
                 discriminator='A',
                 scope='Patch_Discriminator',
                 name='Patch_Discriminator'):
        super().__init__(scope=scope + '_' + discriminator, name=name + '_' + discriminator)

        self.activation = activation
        self.normalization = normalization
        self.f_out = f_out
        self.gamma = gamma
        self.lambd_k = lambd_k
        self.is_training = is_training
        self.loss_name = loss_name
        self.path = os.path.realpath(__file__)


class Patch_Discriminator(IModel):
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
        self.k = tf.placeholder(tf.float32, shape=[], name='k')
        self.kt = np.float32(0.0)
        self.is_training = is_training

    def build_model(self, Y, G, G_pool, G_cyc):
        """
        Build model and create summary

        # Arguments
            Y: 4D-Tensor real image
            G: 4D-Tensor fake image

        # Return
            Tensor of dimension 4D
        """
        self.Y = self.model(Y, is_train=True, collection=0)
        self.G = self.model(G, is_train=True, reuse=True, collection=1)
        self.G_pool = self.model(G_pool, is_train=True, reuse=True, collection=2)
        self.G_cyc = self.model(G_cyc, is_train=True, reuse=True, collection=3)

        return self.Y, self.G, self.G_pool, self.G_cyc

    def model(self, input, is_train=False, reuse=False, collection=0):
        """
        Create generator model

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """

        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            f_out = 64
            net = conv2d(input,
                         f_out=f_out,       #64
                         k_size=4,
                         stride=2,
                         activation=self.activation,
                         reuse=reuse,
                         name='conv_1')

            if collection is 0:
                tf.add_to_collection(self.params.name + '_feature_real', net)
            elif collection is 1:
                tf.add_to_collection(self.params.name + '_feature_fake', net)
            elif collection is 2:
                tf.add_to_collection(self.params.name + '_feature_fake_pool', net)
            else:
                tf.add_to_collection(self.params.name + '_feature_cyc', net)

            net = conv2d(net,
                         f_out=f_out * 2,   #128
                         k_size=4,
                         stride=2,
                         activation=self.activation,
                         normalization=self.normalization,
                         is_training=self.is_training,
                         reuse=reuse,
                         name='conv_2')

            if collection is 0:
                tf.add_to_collection(self.params.name + '_feature_real', net)
            elif collection is 1:
                tf.add_to_collection(self.params.name + '_feature_fake', net)
            elif collection is 2:
                tf.add_to_collection(self.params.name + '_feature_fake_pool', net)
            else:
                tf.add_to_collection(self.params.name + '_feature_cyc', net)

            net = conv2d(net,
                         f_out=f_out * 4,   #256
                         k_size=4,
                         stride=2,
                         activation=self.activation,
                         normalization=self.normalization,
                         is_training=self.is_training,
                         reuse=reuse,
                         name='conv_3')

            if collection is 0:
                tf.add_to_collection(self.params.name + '_feature_real', net)
            elif collection is 1:
                tf.add_to_collection(self.params.name + '_feature_fake', net)
            elif collection is 2:
                tf.add_to_collection(self.params.name + '_feature_fake_pool', net)
            else:
                tf.add_to_collection(self.params.name + '_feature_cyc', net)

            net = conv2d(net,
                         f_out=f_out * 8,   #512
                         k_size=4,
                         stride=1,
                         activation=self.activation,
                         normalization=self.normalization,
                         is_training=self.is_training,
                         reuse=reuse,
                         name='conv_4')

            if collection is 0:
                tf.add_to_collection(self.params.name + '_feature_real', net)
                tf.add_to_collection(self.params.name + '_feature_real_last', net)
            elif collection is 1:
                tf.add_to_collection(self.params.name + '_feature_fake', net)
                tf.add_to_collection(self.params.name + '_feature_fake_last', net)
            elif collection is 2:
                tf.add_to_collection(self.params.name + '_feature_fake_pool', net)
                tf.add_to_collection(self.params.name + '_feature_fake_pool_last', net)
            else:
                tf.add_to_collection(self.params.name + '_feature_cyc', net)
                tf.add_to_collection(self.params.name + '_feature_cyc_last', net)

            net = conv2d(net,
                         f_out=self.params.f_out,
                         k_size=4,
                         stride=1,
                         reuse=reuse,
                         name='conv_5')

        print(' [*] Patch-Discriminator loaded...')
        return net

    def loss_perceptual(self, use_last_layer=True, normalize_loss=True):
        """Retrieve data from the input source and return an object."""
        if not use_last_layer:
            features_real = tf.get_collection(self.params.name + '_feature_real')
            features_fake = tf.get_collection(self.params.name + '_feature_fake')
            features_fake_pool = tf.get_collection(self.params.name + '_feature_fake_pool')
            features_cyc = tf.get_collection(self.params.name + '_feature_cyc')
        else:
            features_real = tf.get_collection(self.params.name + '_feature_real_last')
            features_fake = tf.get_collection(self.params.name + '_feature_fake_last')
            features_fake_pool = tf.get_collection(self.params.name + '_feature_fake_pool_last')
            features_cyc = tf.get_collection(self.params.name + '_feature_cyc_last')

        feature_count = len(features_real)
        perceptual_total_loss_real_fake = 0
        perceptual_total_loss_cyc = 0
        perceptual_total_loss_fake_pool = 0
        for i in range(feature_count):
            if normalize_loss:
                perceptual_total_loss_real_fake += loss_normalization(loss_mse(features_real[i], features_fake[i])) * 1.0 / feature_count
                perceptual_total_loss_fake_pool += loss_normalization(loss_mse(features_real[i], features_fake_pool[i])) * 1.0 / feature_count
                perceptual_total_loss_cyc += loss_normalization(loss_mse(features_real[i], features_cyc[i])) * 1.0 / feature_count
            else:
                perceptual_total_loss_real_fake += loss_mse(features_real[i], features_fake[i]) * 1.0 / feature_count
                perceptual_total_loss_fake_pool += loss_mse(features_real[i], features_fake_pool[i]) * 1.0 / feature_count
                perceptual_total_loss_cyc += loss_mse(features_real[i], features_cyc[i]) * 1.0 / feature_count

        return perceptual_total_loss_real_fake, perceptual_total_loss_cyc, perceptual_total_loss_fake_pool

    def loss_discriminator(self):

        self.loss_D_real = ls_discriminator(self.Y, True)
        self.loss_D_fake = ls_discriminator(self.G, False)
        self.loss_D_fake_pool = ls_discriminator(self.G_pool, False)
        self.loss_G = ls_generator(self.G)

        self.discriminator_loss = 0.5 * self.loss_D_real + 0.5 * self.loss_D_fake_pool
        self.adversarial_loss = self.loss_G

        balance = self.params.gamma * self.loss_D_real - self.loss_G
        self.summary.append(tf.summary.scalar('Balance_' + self.params.name, balance))
        balance_total = self.discriminator_loss - self.adversarial_loss
        self.summary.append(tf.summary.scalar('Balance_total_' + self.params.name, balance_total))
        self.summary.append(tf.summary.scalar('k_' + self.params.name, self.k))
        self.summary.append(tf.summary.scalar('D_real_' + self.params.name, self.loss_D_real))
        self.summary.append(tf.summary.scalar('D_fake_pool_' + self.params.name, self.loss_D_fake_pool))
        self.summary.append(tf.summary.scalar('G_fake_' + self.params.name, self.loss_G))

        self.summary.append(tf.summary.scalar('Adversarial_Loss_' + self.params.name, self.adversarial_loss))
        self.summary.append(tf.summary.scalar('Discriminator_Loss_' + self.params.name, self.discriminator_loss))

        return self.adversarial_loss, self.discriminator_loss

