import time

from tfcore.interfaces.IInferencing import *
from CycleGAN.Generator.generator_unet import *
from CycleGAN.Generator.generator_cyclegan import *


class Inferencer_Params(IInferencer_Params):
    def __init__(self,
                 params_path='',
                 pretrained_generator_dir="../../../pretrained_models/generator_final",
                 domain='AtoB',
                 load=False,
                 ):
        super().__init__()

        self.pretrained_generator_dir = pretrained_generator_dir

        self.normalization_G = 'LN'
        self.use_unet = False
        self.use_recursive_block = False
        self.units = 3
        self.blocks = 3
        self.domain = domain

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

    def load(self, path):
        return super().load(os.path.join(path, "Trainer_Params.json"))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, "Trainer_Params.json"))
        return


class Inferencer(IInferencing):

    def __init__(self, params):
        super().__init__(params)

        self.global_step = tf.Variable(0, trainable=False)

        if self.params.use_unet:
            generator_params_AtoB = Generator_UNet_Params(activation='relu',
                                                          normalization=self.params.normalization_G,
                                                          filter_dim=64,
                                                          use_patches=False,
                                                          generator='AtoB')
            generator_params_AtoB.decay = self.params.decay
            generator_params_AtoB.step_decay = self.params.step_decay
            generator_params_AtoB.beta1 = self.params.beta1
            generator_params_AtoB.learning_rate = self.params.learning_rate_G

            self.generator_AtoB = Generator_UNet_Model(self.sess, generator_params_AtoB, self.global_step)

            generator_params_BtoA = Generator_UNet_Params(activation='relu',
                                                          normalization=self.params.normalization_G,
                                                          filter_dim=64,
                                                          use_patches=False,
                                                          generator='BtoA')
            generator_params_BtoA.decay = self.params.decay
            generator_params_BtoA.step_decay = self.params.step_decay
            generator_params_BtoA.beta1 = self.params.beta1
            generator_params_BtoA.learning_rate = self.params.learning_rate_G

            self.generator_BtoA = Generator_UNet_Model(self.sess, generator_params_BtoA, self.global_step)
        else:
            generator_params_AtoB = Generator_Params(activation='relu',
                                                     normalization=self.params.normalization_G,
                                                     filter_dim=64,
                                                     use_recursice_block=self.params.use_recursive_block,
                                                     units=self.params.units,
                                                     blocks=self.params.blocks,
                                                     is_training=True,
                                                     generator='AtoB')

            self.generator_AtoB = Generator_Model(self.sess, generator_params_AtoB, self.global_step, is_training=False)

            generator_params_BtoA = Generator_Params(activation='relu',
                                                     normalization=self.params.normalization_G,
                                                     filter_dim=64,
                                                     use_recursice_block=self.params.use_recursive_block,
                                                     units=self.params.units,
                                                     blocks=self.params.blocks,
                                                     is_training=True,
                                                     generator='BtoA')

            self.generator_BtoA = Generator_Model(self.sess, generator_params_BtoA, self.global_step, is_training=False)

        self.pretrained_generator_dir = self.params.pretrained_generator_dir

        if self.build_model_inference():
            print(' [*] Build model pass...')

        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] All variables initialized...')

        self.generator_AtoB.load(self.params.pretrained_generator_dir)
        self.generator_BtoA.load(self.params.pretrained_generator_dir)

    def build_model_inference(self):
        with tf.device("/gpu:0"):
            self.inputs = tf.placeholder(tf.float32,
                                         [None, None, None, 3],
                                         name='inputs_A')

            if self.params.domain is 'AtoB':
                self.G = self.generator_AtoB.build_model(self.inputs)
            else:
                self.G = self.generator_BtoA.build_model(self.inputs)

        print('Model loaded...')
        return

    def inference(self, input):

        input = normalize(input).astype(np.float)
        w, h, c = input.shape

        sample = np.asarray(input.reshape(1, w, h, c))

        imageOut = self.G.eval(feed_dict={self.inputs: sample}, session=self.sess)

        imageOut = inormalize(imageOut)

        return imageOut.astype(np.uint8)
