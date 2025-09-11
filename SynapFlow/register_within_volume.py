import os
import glob
import argparse
from tqdm import tqdm

from skimage.transform import resize
from skimage import io
import numpy as np
import itk

def register(
        fixed: itk.Image, moving: itk.Image, reg_type: str
        ) -> "tuple[np.ndarray, itk.TransformixFilter]":
    ndim = fixed.GetImageDimension()

    param_object = itk.ParameterObject.New()
    param_map = param_object.GetDefaultParameterMap(reg_type)
    param_object.AddParameterMap(param_map)

    param_object.SetParameter("MaximumNumberOfIterations", "1000")
    param_object.SetParameter("CenterOfRotation", ['0']*ndim)

    moved, transf_params = itk.elastix_registration_method(
        fixed,
        moving,
        parameter_object=param_object,
        log_to_console=False
        )

    moved = itk.GetArrayFromImage(moved)
    moved[moved < 0] = 0
    moved[moved >= 255] = 255
    moved = np.asarray(moved, dtype=np.uint8)

    return moved, transf_params

def apply_transformation(
        img: itk.Image, transf_params: itk.TransformixFilter) -> np.ndarray:
    moved = itk.transformix_filter(
        img, transf_params)
    moved = itk.GetArrayFromImage(moved)
    moved[moved < 0] = 0
    moved[moved >= 255] = 255
    moved = np.asarray(moved, dtype=np.uint8)

    return moved

def downsample_and_register_itk(
        images: np.ndarray, downsample_factor: float
        ) -> "tuple[np.ndarray, list[itk.TransformixFilter]]":
    orig_w, orig_h = images[0].shape[:2]
    tmp_w, tmp_h = orig_w //downsample_factor, orig_h //downsample_factor

    all_transf = []
    
    for i in tqdm(range(1, len(images)), desc='Registering images'):
        prev_ds = resize(
            images[i-1], (tmp_h, tmp_w), anti_aliasing=True, preserve_range=True
            )
        curr_ds = resize(
            images[i], (tmp_h, tmp_w), anti_aliasing=True, preserve_range=True
            )

        prev_itk_ds = itk.GetImageFromArray(prev_ds.astype(np.float32))
        curr_itk_ds = itk.GetImageFromArray(curr_ds.astype(np.float32))

        prev_itk_ds.SetSpacing([downsample_factor, downsample_factor])
        curr_itk_ds.SetSpacing([downsample_factor, downsample_factor])

        _, transf = register(
            fixed=prev_itk_ds, moving=curr_itk_ds, reg_type='rigid')
        
        transf.SetParameter(0, 'Size', [str(orig_h), str(orig_w)])
        transf.SetParameter(0, 'Spacing', [str(1), str(1)])

        curr_itk = itk.GetImageFromArray(images[i].astype(np.float32))
        out_reg = apply_transformation(curr_itk, transf) # update reference image
        images[i] = out_reg

        all_transf.append(transf)
    
    return images, all_transf

def run_registration(
        input: str, out_dir: str, downsample_factor: float, save_transf: bool, mip_type: str
        ) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mips'), exist_ok=True)

    img_filepaths = sorted(glob.glob(input))
    images = np.array([io.imread(fp) for fp in img_filepaths])

    images, all_transf = downsample_and_register_itk(images, downsample_factor)

    # save transformations using filenames of images. only if save_transf
    if save_transf:
        os.makedirs(os.path.join(out_dir, 'transf'), exist_ok=True)
        for i, tr in enumerate(all_transf):
            transf_fp = os.path.join(
                out_dir, 'transf', os.path.basename(img_filepaths[i+1]) + '.txt'
                )
            tr.WriteParameterFile(tr.GetParameterMap(0), transf_fp)

    # Save registered images
    for i, img in enumerate(images):
        io.imsave(os.path.join(out_dir, os.path.basename(img_filepaths[i])), img)

    # Compute 2D projection of the volume and save
    mip_operator = np.max if mip_type == 'max' else np.mean
    mip_img = mip_operator(images, axis=0).astype(np.uint8)
    io.imsave(os.path.join(out_dir, 'mips', os.path.basename(input).split('*')[0] + '.png'), mip_img)

def run_registration_args(args):
    run_registration(
        args.input,
        args.out_dir,
        args.downsample_factor,
        args.save_transf,
        args.mip_type
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pairwise image registration within volume'
        )
    parser.add_argument('--input', type=str,
        help='Input image volume filename pattern, e.g. /path/to/fov-01_tp-01_*.png') # Not implemented for TIF files yet
    parser.add_argument('--out_dir', type=str,
        help='Output directory to save registered images')
    parser.add_argument('--downsample_factor', type=float, default=4.0,
        help='Downsample factor for registration (default: 4.0)')
    parser.add_argument('--save_transf', action='store_true',
        help='Whether to save the transformation parameters')
    parser.add_argument('--mip_type', type=str, choices=['max', 'mean'],
        default='max',
        help='Whether to use maximum or mean intensity projection (after registration). Default: max.')

    args = parser.parse_args()

    run_registration(
        args.input,
        args.out_dir,
        args.downsample_factor,
        args.save_transf,
        args.mip_type
        )
    
# TODO: enable MIP computation only option (if no registration needed)
# TODO: add code to enable using the transformations to register other channels