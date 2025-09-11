import os, sys
import numpy as np
import argparse
from typing import Union, Tuple


import torch
from PIL import Image

import pump.test_singlescale as pump_main
from pump.post_filter import densify_corres

def compute_dense_corres(
        img1_path: str, 
        img2_path: str, 
        output_path: str, 
        resize: int,
        device: str, 
        hw: Tuple[int, int]
        ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Given a pair of images, compute dense correspondences.
    """
    args = argparse.Namespace(
        img1=img1_path, # fixed image
        img2=img2_path, # moving image
        output=output_path,
        resize=resize,
        device=device,
        backward=device,
        forward=device,
        reciprocal=device,
        post_filter=True,
        desc='PUMP-stytrf',
        levels=99,
        nlpow=1.5,
        border=0.9,
        dtype='float16',
        first_level='torch',
        activation='torch',
        verbose=0,
        dbg=(),
        min_shape=5
        )
        
    pump_main.Main().run_from_args(args)

    corres = np.load(output_path, allow_pickle=True)['corres']
    if len(corres) <= 3:
        print(f'Densification skipped, not enough correspondences in {output_path}')
        return None
    dense_corres = densify_corres(corres, hw)
    np.savez(open(output_path.replace('_sparse', '_dense'),'wb'), corres=dense_corres)

    return corres, dense_corres

def run_compute_dense_corres(
        fixed_image: str,
        moving_image: str,
        out_dir: str, 
        resize: int, 
        device: str
        ) -> None:
    height, width = Image.open(fixed_image).size[:2]
    out_fn = 'fixed=' + fixed_image.split('/')[-1][:-4] + '_moving=' + moving_image.split('/')[-1][:-4]
    out_fp = os.path.join(out_dir, out_fn + '_sparse.npy')

    compute_dense_corres(
        fixed_image,
        moving_image,
        out_fp,
        resize,
        device=device,
        hw=(height, width)
        )
    
def run_compute_dense_corres_args(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    run_compute_dense_corres(
        args.fixed,
        args.moving,
        args.out_dir,
        args.resize,
        device
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute dense correspondences between two images'
        )
    parser.add_argument('-fi', '--fixed', type=str,
                        help='file path to fixed image')
    parser.add_argument('-mo', '--moving', type=str,
                        help='file path to moving image')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='directory to save output')
    parser.add_argument('-r', '--resize', type=int, default=0,
                        help='resize images to this value')
    args = parser.parse_args()

    run_compute_dense_corres_args(args)