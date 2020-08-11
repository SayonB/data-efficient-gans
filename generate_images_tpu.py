import argparse
import os
import pathlib
from pathlib import Path
from pprint import pprint as pp

import numpy as np
import tensorflow as tf
import tqdm
from tqdm import tqdm

import dnnlib
import tflex
from dnnlib import EasyDict
from training import misc
from training.networks_stylegan2 import *

try:
    from StringIO import StringIO as BytesIO  # for Python 2
except ImportError:
    from io import BytesIO  # for Python 3


def rand_latent(n, seed=None):
    if seed is not None:
        if seed < 0:
            seed = 2*32 - seed
        np.random.seed(seed)
    result = np.random.randn(n, *G.input_shape[1:])
    if seed is not None:
        np.random.seed()
    return result


def tfinit():
    tflib.run(tf.global_variables_initializer())


def load_checkpoint(path, checkpoint_num=None):
    if checkpoint_num is None:
        ckpt = tf.train.latest_checkpoint(path)
    else:
        ckpt = os.path.join(path, f'model.ckpt-{checkpoint_num}')
    assert ckpt is not None
    print('Loading checkpoint ' + ckpt)
    saver.restore(sess, ckpt)
    return ckpt


def get_checkpoint(path):
    ckpt = tf.train.latest_checkpoint(path)
    return ckpt


def get_grid_size(n):
    gw = 1
    gh = 1
    i = 0
    while gw*gh < n:
        if i % 2 == 0:
            gw += 1
        else:
            gh += 1
        i += 1
    return (gw, gh)


def gen_images(latents, truncation_psi_val, outfile=None, display=False, labels=None, randomize_noise=False, is_validation=True, network=None, numpy=False):
    if outfile:
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    
    if network is None:
        network = Gs
    n = latents.shape[0]
    grid_size = get_grid_size(n)
    drange_net = [-1, 1]
    with tflex.device('/gpu:0'):
        result = network.run(latents, labels, truncation_psi_val=truncation_psi_val, is_validation=is_validation, randomize_noise=randomize_noise,
                             minibatch_size=sched.minibatch_gpu)
        result = result[:, 0:3, :, :]
        img = misc.convert_to_pil_image(
            misc.create_image_grid(result, grid_size), drange_net)
        if outfile is not None:
            img.save(outfile)
        if display:
            f = BytesIO()
            img.save(f, 'png')
            IPython.display.display(IPython.display.Image(data=f.getvalue()))
    return result if numpy else img


def grab(save_dir, i, n=1, latents=None, **kwargs):
    if latents is None:
        latents = rand_latent(n, seed=i)
    gw, gh = get_grid_size(latents.shape[0])
    outfile = str(save_dir/str(i)) + '.png'
    return gen_images(latents, outfile=outfile, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleGAN2 TPU Generator')

    parser.add_argument('--model_dir', type=str, action='store',
                        help='Location of the checkpoint files')
    parser.add_argument('--save_dir', type=str, action='store',
                        help='Location of the directory to save images in')
    parser.add_argument('--truncation_psi', type=float, action='store',
                        help='Truncation psi (default: %(default)s)', default=0.7)
    parser.add_argument('--num_samples', type=int, action='store',
                        help='Number of samples to generate (default: %(default)s)',
                        default=1)
    parser.add_argument('--checkpoint_num', type=int, action='store',
                        help='The checkpoint to use to generate the images. The default is the latest checkpoint in model_dir', default=None)

    args = parser.parse_args()

    # --- set resolution and label size here:
    label_size = 0
    resolution = 512
    fmap_base = (int(os.environ['FMAP_BASE'])
                 if 'FMAP_BASE' in os.environ else 16) << 10
    num_channels = 3
    channel = 3
    count = 1
    grid_image_size = int(
        os.environ['GRID_SIZE']) if 'GRID_SIZE' in os.environ else 9
    # ------------------------
    print('------------------')
    print('Initializing model')
    print('------------------')

    # set up model
    dnnlib.tflib.init_tf()

    sess = tf.get_default_session()
    sess.list_devices()

    cores = tflex.get_cores()
    tflex.set_override_cores(cores)

    # Options for training loop.
    train = EasyDict(run_func_name='training.training_loop.training_loop')
    # Options for generator network.
    G_args = EasyDict(func_name='training.networks_stylegan2.G_main')
    # Options for discriminator network.
    D_args = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')
    # Options for generator optimizer.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for discriminator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    # Options for loss.
    loss_args = EasyDict(func_name='training.loss.ns_DiffAugment_r1')
    # Options for TrainingSchedule.
    sched = EasyDict()
    # Options for setup_snapshot_image_grid().
    grid = EasyDict(size='4k', layout='random')
    # Options for dnnlib.submit_run().
    sc = dnnlib.SubmitConfig()
    tf_config = {'rnd.np_random_seed': 1000}
    label_dtype = np.int64
    sched.minibatch_gpu = 1

    if 'G' not in globals():
        with tflex.device('/gpu:0'):
            G = tflib.Network('G', num_channels=num_channels, resolution=resolution,
                              label_size=label_size, **G_args)
            G.print_layers()
            Gs, Gs_finalize = G.clone2('Gs')
            Gs_finalize()
            D = tflib.Network('D', num_channels=num_channels, resolution=resolution,
                              label_size=label_size, **D_args)
            D.print_layers()

    grid_size = (2, 2)
    gw, gh = grid_size
    gn = np.prod(grid_size)
    grid_latents = rand_latent(gn, seed=-1)
    grid_labels = np.zeros([gw * gh, label_size], dtype=label_dtype)

    tfinit()
    print('-----------------')
    print('Initialized model')
    print('-----------------')
    # ----------------------
    # Load checkpoint
    print('Loading checkpoint')
    saver = tf.train.Saver()

    # ------------------------
    model_ckpt = load_checkpoint(args.model_dir, args.checkpoint_num)
    model_name = model_ckpt.split('/')[4]
    model_num = model_ckpt.split('-')[-1]
    print('Loaded model', model_name)

    tflex.state.noisy = False
    # ---------------------
    
    save_dir = Path(args.save_dir)/model_num
    # generate samples
    for i in tqdm(list(range(args.num_samples))):
        grab(save_dir, i=i,
             truncation_psi_val=args.truncation_psi)  # modify seed
