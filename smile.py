import os
import shutil
import pickle
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import cv2
import sys

if __name__ == "__main__":

    if len(sys.argv)>1:
        ENC_DIR = sys.argv[1]
    else:
        ENC_DIR = 'enc'
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    else:
        OUTPUT_DIR = os.path.join('..', 'output')

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    URL_FFHQ_mirror = 'https://drive.google.com/uc?id=19B138TWKeOs-JIol0_K-CCCDMYXbK5bk'

    tflib.init_tf()
    try:
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    except:
        with dnnlib.util.open_url(URL_FFHQ_mirror, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)


    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)


    def generate_image(latent_vector):
        latent_vector = latent_vector.reshape((1, 18, 512))
        generator.set_dlatents(latent_vector)
        return generator.generate_images()[0]


    def move_and_show(latent_vector, direction, coeffs, out_name):
        vid_name = os.path.join(OUTPUT_DIR, out_name.replace('.npy', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video = cv2.VideoWriter(vid_name, fourcc, 30, (1024, 1024))
        gen = {}
        for i, coeff in enumerate(coeffs):
            new_latent_vector = latent_vector.copy()
            new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
            if coeff not in gen:
                gen[coeff] = generate_image(new_latent_vector)
            video.write(gen[coeff][..., ::-1])
        video.release()
        print('finished '+vid_name)

    smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
    coeffs = np.concatenate([np.arange(0, 2, .02), np.arange(2, -2, -.02), np.arange(-2, 0, .02)])
    for file in os.listdir(ENC_DIR):
        img = np.load(os.path.join(ENC_DIR, file))
        move_and_show(img, smile_direction, coeffs, file)