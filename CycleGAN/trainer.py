import time
import math
from tfcore.core.loss import *
from tfcore.interfaces.ITraining import *
from tfcore.utilities.image_pool import ImagePool
from tfcore.core.normalization import SPECTRAL_NORM_UPDATE_OPS
from tfcore.utilities.utils import get_patches
from CycleGAN.Generator.generator_unet import Generator_UNet_Model, Generator_UNet_Params
from CycleGAN.Generator.generator_cyclegan import Generator_Model, Generator_Params
from CycleGAN.Discriminator.patch_discriminator import Patch_Discriminator, Patch_Discriminator_Params


class Trainer_Params(ITrainer_Params):
    """
    Parameter-Class for Example-Trainer
    """

    def __init__(self,
                 image_size,
                 params_path='',
                 load=True,
                 ):
        """ Constructor

        # Arguments
            image_size: Image size (int)
            params_path: Parameter-Path for loading and saving (str)
            load: Load parameter (boolean)
        """
        super().__init__()

        self.image_size = image_size
        self.epoch = 100
        self.batch_size = 1
        self.decay = 0.75
        self.step_decay = 3125
        self.beta1 = 0.5
        self.learning_rate_G = 0.0002
        self.learning_rate_D = 0.0002
        self.use_tensorboard = True
        self.gpus = [0]

        self.use_NN = True
        self.pool_size = 50
        self.normalization_G = 'LN'
        self.normalization_D = 'IN'

        # U-Net Params
        self.use_unet = False
        self.depth = 5

        # ResNet Params
        self.use_recursive_block = False
        self.units = 3
        self.blocks = 3

        # Loss Params
        self.use_perceptual_loss = True
        self.use_last_layer_only = True
        self.nomalize_perceptual_loss = False

        self.use_pretrained_generator = True
        self.pretrained_generator_dir = "../../../../pretrained_models/generator/"

        self.experiment_name = "CycleGAN"
        self.checkpoint_restore_dir = ''
        self.sample_dir = 'samples'
        self.load_checkpoint = False

        self.use_validation_set = True
        self.evals_per_iteration = 25
        self.save_checkpoint = False

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

        self.root_dir = "../../../Results/Local"

    def load(self, path):
        """ Load Parameter

        # Arguments
            path: Path of json-file
        # Return
            Parameter class
        """
        return super().load(os.path.join(path, "Trainer_Params"))

    def save(self, path):
        """ Save parameter as json-file

        # Arguments
            path: Path to save
        """
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, "Trainer_Params"))
        return


class CycleGAN_Trainer(ITrainer):
    """ A example class to train a generator neural-network
    """

    def __init__(self, trainer_params):
        """
        # Arguments
            trainer_params: Parameter from class Example_Trainer_Params
        """

        #   Initialize the abstract Class ITrainer
        super().__init__(trainer_params)

        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        if self.params.use_unet:
            generator_params_AtoB = Generator_UNet_Params(activation='relu',
                                                          normalization=self.params.normalization_G,
                                                          filter_dim=64,
                                                          depth=self.params.depth,
                                                          use_NN=self.params.use_NN,
                                                          patch_size=self.params.image_size,
                                                          generator='AtoB')
            generator_params_AtoB.decay = self.params.decay
            generator_params_AtoB.step_decay = self.params.step_decay
            generator_params_AtoB.beta1 = self.params.beta1
            generator_params_AtoB.learning_rate = self.params.learning_rate_G

            self.generator_AtoB = Generator_UNet_Model(self.sess, generator_params_AtoB, self.global_step, self.is_training)

            generator_params_BtoA = Generator_UNet_Params(activation='relu',
                                                          normalization=self.params.normalization_G,
                                                          filter_dim=64,
                                                          depth=self.params.depth,
                                                          use_NN=self.params.use_NN,
                                                          patch_size=self.params.image_size,
                                                          generator='BtoA')
            generator_params_BtoA.decay = self.params.decay
            generator_params_BtoA.step_decay = self.params.step_decay
            generator_params_BtoA.beta1 = self.params.beta1
            generator_params_BtoA.learning_rate = self.params.learning_rate_G

            self.generator_BtoA = Generator_UNet_Model(self.sess, generator_params_BtoA, self.global_step, self.is_training)
        else:
            generator_params_AtoB = Generator_Params(activation='relu',
                                                     normalization=self.params.normalization_G,
                                                     filter_dim=64,
                                                     use_NN=self.params.use_NN,
                                                     is_training=True,
                                                     use_recursice_block=self.params.use_recursive_block,
                                                     units=self.params.units,
                                                     blocks=self.params.blocks,
                                                     generator='AtoB')
            generator_params_AtoB.decay = self.params.decay
            generator_params_AtoB.step_decay = self.params.step_decay
            generator_params_AtoB.beta1 = self.params.beta1
            generator_params_AtoB.learning_rate = self.params.learning_rate_G

            self.generator_AtoB = Generator_Model(self.sess, generator_params_AtoB, self.global_step, self.is_training)

            generator_params_BtoA = Generator_Params(activation='relu',
                                                     normalization=self.params.normalization_G,
                                                     filter_dim=64,
                                                     use_NN=self.params.use_NN,
                                                     is_training=True,
                                                     use_recursice_block=self.params.use_recursive_block,
                                                     units=self.params.units,
                                                     blocks=self.params.blocks,
                                                     generator='BtoA')
            generator_params_BtoA.decay = self.params.decay
            generator_params_BtoA.step_decay = self.params.step_decay
            generator_params_BtoA.beta1 = self.params.beta1
            generator_params_BtoA.learning_rate = self.params.learning_rate_G

            self.generator_BtoA = Generator_Model(self.sess, generator_params_BtoA, self.global_step, self.is_training)

        discriminator_params_A = Patch_Discriminator_Params(activation='lrelu',
                                                            normalization=self.params.normalization_D,
                                                            f_out=1,
                                                            discriminator='A')
        discriminator_params_A.decay = self.params.decay
        discriminator_params_A.step_decay = self.params.step_decay
        discriminator_params_A.beta1 = self.params.beta1
        discriminator_params_A.learning_rate = self.params.learning_rate_D

        self.discriminator_A = Patch_Discriminator(self.sess, discriminator_params_A, self.global_step, self.is_training)

        discriminator_params_B = Patch_Discriminator_Params(activation='lrelu',
                                                            normalization=self.params.normalization_D,
                                                            f_out=1,
                                                            discriminator='B')
        discriminator_params_B.decay = self.params.decay
        discriminator_params_B.step_decay = self.params.step_decay
        discriminator_params_B.beta1 = self.params.beta1
        discriminator_params_B.learning_rate = self.params.learning_rate_D

        self.discriminator_B = Patch_Discriminator(self.sess, discriminator_params_B, self.global_step, self.is_training)

        #   Create the directorys for logs, checkpoints and samples
        self.prepare_directorys()
        #   Save the hole dl_core library as zip
        save_experiment(self.checkpoint_dir)
        #   Save the Trainer_Params as json
        self.params.save(self.checkpoint_dir)

        #   Placeholder for input x
        self.all_A = tf.placeholder(tf.float32,
                                    [self.batch_size_total,
                                     self.params.image_size,
                                     self.params.image_size, 3],
                                    name='all_A')

        #   Placeholder for ground-truth Y
        self.all_B = tf.placeholder(tf.float32,
                                    [self.batch_size_total,
                                     self.params.image_size,
                                     self.params.image_size, 3],
                                    name='all_B')

        self.fake_pool_BtoA = tf.placeholder(tf.float32, [None,
                                                          None,
                                                          None,
                                                          3], name="fake_pool_A")
        self.fake_pool_AtoB = tf.placeholder(tf.float32, [None,
                                                          None,
                                                          None,
                                                          3], name="fake_pool_B")

        self.input_A = None
        self.input_B = None
        self.image_A = None
        self.image_B = None

        self.pool_A = ImagePool(self.params.pool_size)
        self.pool_B = ImagePool(self.params.pool_size)

        #   Build Pipeline
        self.build_pipeline()

        #   Initialize all variables
        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] All variables initialized...')

        self.saver = tf.train.Saver()

        #   Load pre-trained model
        if self.params.use_pretrained_generator or self.params.new is False:
            self.generator_AtoB.load(self.params.pretrained_generator_dir)
            self.generator_BtoA.load(self.params.pretrained_generator_dir)

        #   Load checkpoint
        if self.params.load_checkpoint:
            load(self.sess, self.params.checkpoint_restore_dir)

        self.update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)

        return

    def prepare_directorys(self):
        """ Prepare the directorys for logs, samples and checkpoints
        """
        self.model_dir = "%s__%s" % (
            self.generator_AtoB.params.name, self.batch_size)
        self.sample_dir = os.path.join(self.params.root_dir, 'samples',
                                       self.params.experiment_name + '_GPU' +
                                       str(self.params.gpus),
                                       self.sample_dir + '_' + self.model_dir)
        self.checkpoint_dir = os.path.join(self.params.root_dir, 'checkpoints',
                                           self.params.experiment_name + '_GPU' +
                                           str(self.params.gpus),
                                           self.checkpoint_dir)
        self.log_dir = os.path.join(self.params.root_dir, 'logs',
                                    self.params.experiment_name + '_GPU' +
                                    str(self.params.gpus))

        if self.params.new:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            if os.path.exists(self.sample_dir):
                shutil.rmtree(self.sample_dir)
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def set_validation_set(self, batch_valid_X, batch_valid_Y):
        """ Set the validation set for x and Y which same batch-size like training-examples

        # Arguments
            batch_valid_X: samples for x
            batch_valid_Y: samples for Y
        """
        self.image_A = normalize(batch_valid_X)
        self.image_B = normalize(batch_valid_Y)

        save_images(self.image_A, [int(math.sqrt(self.batch_size)),
                                   int(math.sqrt(self.batch_size))],
                    self.sample_dir + '/image_A.png')
        save_images(self.image_B, [int(math.sqrt(self.batch_size)),
                                   int(math.sqrt(self.batch_size))],
                    self.sample_dir + '/image_B.png', normalized=True)

        print(' [*] Image_A ' + str(self.image_A.shape))
        print(' [*] Image_B ' + str(self.image_B.shape))

    def validate(self, epoch, iteration, idx):
        """ Validate the validation-set

        # Arguments
            epoch:   Current epoch
            iteration: Current interation
            idx: Index in current epoch
        """

        #   Validate Samples
        samples_A, g_loss_val, g_summery_A = self.sess.run([self.sample_AtoB,
                                                            self.generator_AtoB.total_loss,
                                                            self.summary_val],
                                                           feed_dict={self.all_A: self.image_A,
                                                                      self.all_B: self.image_B,
                                                                      self.fake_pool_BtoA: self.image_A,
                                                                      self.fake_pool_AtoB: self.image_B,
                                                                      self.discriminator_A.k: self.discriminator_A.kt,
                                                                      self.discriminator_B.k: self.discriminator_B.kt,
                                                                      self.epoch: epoch,
                                                                      self.generator_AtoB.learning_rate: self.params.learning_rate_G,
                                                                      self.is_training: False})
        samples_B, g_loss_val, g_summery_B = self.sess.run([self.sample_BtoA,
                                                            self.generator_BtoA.total_loss,
                                                            self.summary_val],
                                                           feed_dict={self.all_A: self.image_A,
                                                                      self.all_B: self.image_B,
                                                                      self.fake_pool_BtoA: self.image_A,
                                                                      self.fake_pool_AtoB: self.image_B,
                                                                      self.discriminator_A.k: self.discriminator_A.kt,
                                                                      self.discriminator_B.k: self.discriminator_B.kt,
                                                                      self.epoch: epoch,
                                                                      self.generator_BtoA.learning_rate: self.params.learning_rate_G,
                                                                      self.is_training: False})

        self.writer.add_summary(g_summery_A, iteration)
        self.writer.add_summary(g_summery_B, iteration)

        if iteration == 0:
            _, g_loss_val, g_summery_A = self.sess.run([self.sample_AtoB,
                                                        self.generator_AtoB.total_loss,
                                                        self.summary_vis_one],
                                                       feed_dict={self.all_A: self.image_A,
                                                                  self.all_B: self.image_B,
                                                                  self.fake_pool_BtoA: self.image_A,
                                                                  self.fake_pool_AtoB: self.image_B,
                                                                  self.discriminator_A.k: self.discriminator_A.kt,
                                                                  self.discriminator_B.k: self.discriminator_B.kt,
                                                                  self.epoch: epoch,
                                                                  self.is_training: False})

            _, g_loss_val, g_summery_B = self.sess.run([self.sample_BtoA,
                                                        self.generator_BtoA.total_loss,
                                                        self.summary_vis_one],
                                                       feed_dict={self.all_A: self.image_A,
                                                                  self.all_B: self.image_B,
                                                                  self.fake_pool_BtoA: self.image_A,
                                                                  self.fake_pool_AtoB: self.image_B,
                                                                  self.discriminator_A.k: self.discriminator_A.kt,
                                                                  self.discriminator_B.k: self.discriminator_B.kt,
                                                                  self.epoch: epoch,
                                                                  self.is_training: False})

            self.writer.add_summary(g_summery_A, iteration)
            self.writer.add_summary(g_summery_B, iteration)

        #   Save samples as png
        save_images(samples_A,
                    [int(math.sqrt(self.batch_size)), int(math.sqrt(self.batch_size))],
                    self.sample_dir + '/sample_A_%s_%s_%s.png' % (epoch, idx, iteration),
                    normalized=True)

        save_images(samples_B,
                    [int(math.sqrt(self.batch_size)), int(math.sqrt(self.batch_size))],
                    self.sample_dir + '/sample_B_%s_%s_%s.png' % (epoch, idx, iteration),
                    normalized=True)

        print("[Sample] g_loss: %.8f" % g_loss_val)

    def make_summarys(self, gradient_list):
        """ Calculate some metrics and add it to the summery for the validation-set

        # Arguments
            gradient_list: Gradients to store in log-file as histogram
        """

        A = get_patches(self.A, patch_size=self.params.image_size)
        B = get_patches(self.B, patch_size=self.params.image_size)

        self.summaries.append(tf.summary.scalar("G_loss_A", self.generator_AtoB.total_loss))
        self.summaries.append(tf.summary.scalar("G_loss_B", self.generator_BtoA.total_loss))
        self.summaries_val.append(tf.summary.scalar("valid_total_loss_A", self.generator_AtoB.total_loss))
        self.summaries_val.append(tf.summary.scalar("valid_total_loss_B", self.generator_BtoA.total_loss))
        self.summaries_val.append(tf.summary.scalar("valid_MAE_A", loss_mae(A, self.AtoBtoA)))
        self.summaries_val.append(tf.summary.scalar("valid_MAE_B", loss_mae(B, self.BtoAtoB)))

        super().make_summarys(gradient_list)

    def set_losses(self,
                   A, B,
                   Cyc_AtoBtoA, Cyc_BtoAtoB):
        """  Generaotion of the loss-function for the CycleGAN

        Paper: https://arxiv.org/pdf/1703.10593.pdf

        # Arguments
            A: Image A
            B: Image B
            Cyc_A: Rekunstruction of A -> B -> A'.
            Cyc_B: Rekunstruction of B -> A -> B'.
            D_real_A: Discriminator A output for real image.
            D_fake_A: Discriminator A output for fake image.
            D_fake_pool_A: Discriminator A output for fake_pool image.
            D_real_B: Discriminator B output for real image.
            D_fake_B: Discriminator B output for fake image.
            D_fake_pool_B: Discriminator B output for fake_pool image.
        """
        """ Concatenate different loss-functions to a total_loss

        # Arguments
            G: Generator-Output G
            Y: Ground-Truth image Y
        """

        self.generator_BtoA.total_loss, self.discriminator_A.total_loss = self.discriminator_A.loss_discriminator()
        self.generator_AtoB.total_loss, self.discriminator_B.total_loss = self.discriminator_B.loss_discriminator()

        # Cycle-Loss

        A = get_patches(A, patch_size=self.params.image_size)
        B = get_patches(B, patch_size=self.params.image_size)

        if self.params.nomalize_perceptual_loss:
            cyc_loss_A = loss_normalization(loss_mae(A, Cyc_AtoBtoA)) * 0.5
            cyc_loss_B = loss_normalization(loss_mae(B, Cyc_BtoAtoB)) * 0.5
        else:
            cyc_loss_A = loss_mae(A, Cyc_AtoBtoA)
            cyc_loss_B = loss_mae(B, Cyc_BtoAtoB)

        cyc_loss = (cyc_loss_A + cyc_loss_B) * 10.0

        if self.params.use_perceptual_loss:
            perceptual_loss_A, perceptual_loss_cyc_A, _ = self.discriminator_A.loss_perceptual(normalize_loss=self.params.nomalize_perceptual_loss)
            perceptual_loss_B, perceptual_loss_cyc_B, _ = self.discriminator_B.loss_perceptual(normalize_loss=self.params.nomalize_perceptual_loss)

            if self.params.nomalize_perceptual_loss:
                cyc_loss += (perceptual_loss_cyc_A + perceptual_loss_cyc_B) * 0.5
            else:
                cyc_loss += (perceptual_loss_cyc_A + perceptual_loss_cyc_B) * 1.0

            self.summaries.append(tf.summary.scalar("perceptual_loss_cyc_A", perceptual_loss_cyc_A))
            self.summaries.append(tf.summary.scalar("perceptual_loss_cyc_B", perceptual_loss_cyc_B))

        self.generator_AtoB.total_loss += cyc_loss
        self.generator_BtoA.total_loss += cyc_loss

        self.summaries.append(tf.summary.scalar("Cycle_loss", cyc_loss))
        self.summaries.append(tf.summary.scalar("Cycle_loss_A", cyc_loss_A))
        self.summaries.append(tf.summary.scalar("Cycle_loss_B", cyc_loss_B))

        self.g_loss = self.generator_AtoB.total_loss + self.generator_BtoA.total_loss
        self.d_loss = self.discriminator_A.total_loss + self.discriminator_B.total_loss

    def build_model(self, tower_id):
        """ Build models for CycleGAN

        Paper: Paper: https://arxiv.org/pdf/1703.10593.pdf

        # Arguments
            tower_id: Tower-ID
        # Return
            List of all existing models witch should trained
        """

        #   Split the total batch by the gpu-count
        A = self.all_A[tower_id * self.batch_size:(tower_id + 1) * self.batch_size, :]
        B = self.all_B[tower_id * self.batch_size:(tower_id + 1) * self.batch_size, :]

        #   Create generator model
        G_AtoB = self.generator_AtoB.build_model(A)
        G_BtoA = self.generator_BtoA.build_model(B)

        Cyc_AtoBtoA = self.generator_BtoA.build_model(G_AtoB, reuse=True)
        Cyc_BtoAtoB = self.generator_AtoB.build_model(G_BtoA, reuse=True)

        self.sample_AtoB = self.generator_AtoB.build_model(A, reuse=True)
        self.sample_BtoA = self.generator_BtoA.build_model(B, reuse=True)

        self.sample_AtoBtoA = self.generator_BtoA.build_model(self.sample_AtoB, reuse=True)
        self.sample_BtoAtoB = self.generator_AtoB.build_model(self.sample_BtoA, reuse=True)

        self.discriminator_A.build_model(A, G_BtoA, self.fake_pool_BtoA, Cyc_AtoBtoA)
        self.discriminator_B.build_model(B, G_AtoB, self.fake_pool_AtoB, Cyc_BtoAtoB)

        self.set_losses(A, B,
                        Cyc_AtoBtoA, Cyc_BtoAtoB)

        self.A = A
        self.AtoBtoA = Cyc_AtoBtoA

        self.B = B
        self.BtoAtoB = Cyc_BtoAtoB

        self.summaries_vis_one.append(tf.summary.image('A', A, max_outputs=1))
        self.summaries_vis_one.append(tf.summary.image('B', B, max_outputs=1))
        self.summaries_val.append(tf.summary.image('B-A', self.sample_BtoA, max_outputs=1))
        self.summaries_val.append(tf.summary.image('A-B', self.sample_AtoB, max_outputs=1))
        self.summaries_val.append(tf.summary.image('A-B-A', self.sample_AtoBtoA, max_outputs=1))
        self.summaries_val.append(tf.summary.image('B-A-B', self.sample_BtoAtoB, max_outputs=1))

        #   Append all models with should be optimized
        model_list = []
        model_list.append(self.generator_AtoB)
        model_list.append(self.generator_BtoA)
        model_list.append(self.discriminator_A)
        model_list.append(self.discriminator_B)

        for model in model_list:
            self.summaries_val.extend(model.summary_val)
            self.summaries_vis.extend(model.summary_vis)
            self.summaries_vis_one.extend(model.summary_vis_one)
            self.summaries.extend(model.summary)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_optim = tf.train.AdamOptimizer(self.discriminator_A.learning_rate, beta1=self.params.beta1).minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.generator_AtoB.learning_rate, beta1=self.params.beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.summaries.append(tf.summary.scalar("LR_Discriminator", self.discriminator_A.learning_rate))
        self.summaries.append(tf.summary.scalar("LR_Generator", self.generator_AtoB.learning_rate))

        return []  # model_list

    def train_online(self, batch_X, batch_Y, epoch=0, counter=1, idx=0, batch_total=0):
        """ Training, validating and saving of the generator model

        # Arguments
            batch_X: Training-Examples for input x
            batch_Y: Training-Examples for ground-truth Y
            epoch: Current epoch
            counter: Current iteration
            idx: Current batch
            batch_total: Total batch size
        """
        start_time = time.time()

        # self.params.learning_rate_G = self.params.learning_rate_D = self.generator_AtoB.crl.get_learning_rate(counter)

        #   Normalize input images between -1 and 1
        self.input_A = normalize(batch_X, normalization_type='tanh')
        self.input_B = normalize(batch_Y, normalization_type='tanh')

        #   Validate after N iterations
        if epoch < 2:
            if np.mod(counter, self.params.evals_per_iteration) == 0:
                self.validate(epoch, counter, idx)
        else:
            if np.mod(counter, batch_total) == 0:
                self.validate(epoch, counter, idx)

        # Optimize Generator

        fake_BtoA, fake_AtoB = self.sess.run([self.generator_BtoA.G_A, self.generator_AtoB.G_A],
                                             feed_dict={self.all_A: self.input_A,
                                                        self.all_B: self.input_B,
                                                        self.is_training: True})
        feed_dict = {self.all_A: self.input_A,
                     self.all_B: self.input_B,
                     self.fake_pool_BtoA: self.pool_A.query(fake_BtoA),
                     self.fake_pool_AtoB: self.pool_A.query(fake_AtoB),
                     self.epoch: epoch,
                     self.generator_AtoB.learning_rate: self.params.learning_rate_G,
                     self.generator_BtoA.learning_rate: self.params.learning_rate_G,
                     self.discriminator_A.learning_rate: self.params.learning_rate_D,
                     self.discriminator_B.learning_rate: self.params.learning_rate_D,
                     self.discriminator_A.k: self.discriminator_A.kt,
                     self.discriminator_B.k: self.discriminator_B.kt,
                     self.is_training: True}

        _, g_loss, summary_G = self.sess.run([self.g_optim, self.g_loss, self.summary],
                                             feed_dict=feed_dict)

        self.writer.add_summary(summary_G, counter)

        _, d_loss, \
        y_energie_A, g_energie_A, \
        y_energie_B, g_energie_B, \
        summary_D = self.sess.run([self.d_optim, self.d_loss,
                                   self.discriminator_A.discriminator_loss, self.discriminator_A.adversarial_loss,
                                   self.discriminator_B.discriminator_loss, self.discriminator_B.adversarial_loss,
                                   self.summary],
                                  feed_dict=feed_dict)

        self.discriminator_A.kt = np.maximum(np.minimum(1.0, self.discriminator_A.kt + self.discriminator_A.params.lambd_k *
                                                        (self.discriminator_A.params.gamma * y_energie_A - g_energie_A)), -1.0)
        self.discriminator_B.kt = np.maximum(np.minimum(1.0, self.discriminator_B.kt + self.discriminator_B.params.lambd_k *
                                                        (self.discriminator_B.params.gamma * y_energie_B - g_energie_B)), -1.0)

        self.writer.add_summary(summary_D, counter)

        print("Train CycleGAN: Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, d_loss: %.8f"
              % (epoch, idx, batch_total, time.time() - start_time, g_loss, d_loss))

        for update_op in self.update_ops:
            self.sess.run(update_op)

        #   Save model and checkpoint
        if np.mod(counter + 1, int(batch_total / 2)) == 0:
            self.generator_AtoB.save(self.sess, self.checkpoint_dir, self.global_step)
            self.generator_BtoA.save(self.sess, self.checkpoint_dir, self.global_step)
            self.discriminator_A.save(self.sess, self.checkpoint_dir, self.global_step)
            self.discriminator_B.save(self.sess, self.checkpoint_dir, self.global_step)
