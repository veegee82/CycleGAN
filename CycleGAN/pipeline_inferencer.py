from CycleGAN.inferencer import Inferencer, Inferencer_Params
from tfcore.interfaces.IPipeline_Inferencer import IPipeline_Inferencer_Params, IPipeline_Inferencer
from tfcore.utilities.preprocessing import Preprocessing
import gflags
import os
import sys
import imageio


def get_filename(idx, filename='', decimals=5):
    for n in range(decimals, -1, -1):
        if idx < pow(10, n):
            filename += '0'
        else:
            filename += str(idx)
            break
    return filename + '.png'

class Pipeline_Inferencer_Params(IPipeline_Inferencer_Params):

    def __init__(self,
                 data_dir,
                 data_dir_x=''):
        super().__init__(data_dir=data_dir, data_dir_x=data_dir_x)


class Pipeline_Inferencer(IPipeline_Inferencer):

    def __init__(self, inferencer, params, pre_processing):
        super().__init__(inferencer, params, pre_processing)


# flags = tf.app.flags
flags = gflags.FLAGS
gflags.DEFINE_string("config_path", '', "Path for config files")
gflags.DEFINE_string("dataset", "../../../Data/horse2zebra/trainB", "Dataset path")
gflags.DEFINE_string("outdir", "../../../Data/Results/Horse2Zebra_AtoB", "Output path")
gflags.DEFINE_string("model_dir", "../../../../pretrained_models/generator_final/Horse2Zebra_LN/", "Model directory")

def main():
    flags(sys.argv)

    model_params = Inferencer_Params(pretrained_generator_dir=flags.model_dir,
                                     domain='AtoB')
    model_inferencer = Inferencer(model_params)

    pre_processing = Preprocessing()
    pre_processing.add_function_xy(Preprocessing.ToRGB().function)

    pipeline_params = Pipeline_Inferencer_Params(data_dir=flags.dataset,
                                                 data_dir_x=flags.dataset)
    pipeline = Pipeline_Inferencer(inferencer=model_inferencer, params=pipeline_params, pre_processing=pre_processing)

    count = 0
    first_pass = True
    while first_pass or img_out is not None:
        if first_pass:
            first_pass = False
            if not os.path.exists(flags.outdir):
                os.makedirs(flags.outdir)

        img_out = pipeline.run()
        if img_out is not None:
            filename = get_filename(count, 'image_')
            imageio.imwrite(os.path.join(flags.outdir, filename), img_out)
            print(' [*] save file ' + filename)
        count += 1


if __name__ == "__main__":
    main()
