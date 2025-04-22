"""\
This script contains helper functions to render LAMMPS dump files with Ovito
"""

import os
import numpy as np

import imageio.v3 as iio
from pygifsicle import optimize
import ovito
from ovito.io import import_file
from ovito.vis import *
from ovito import dataset

def render_dumps(img_size: tuple = (640, 480),dump_path: str = os.getcwd(),orient_name = ['top', 'front', 'left', 'perspective'],gif_path = None) -> None:
    '''
    This function renders the desired views (to chose in ['top', 'front', 'left', 'perspective']) of the provided dump file(s) as images (pngs) or animations (gifs).
    '''

    if gif_path is None:
        gif_path = dump_path

    # Initialize the ovito viewports with the standard camera directions
    vp_top = Viewport(type=Viewport.Type.Top, camera_dir=(0, 0, -1))
    vp_front = Viewport(type=Viewport.Type.Front, camera_dir=(0, 1, 0))
    vp_left = Viewport(type=Viewport.Type.Left, camera_dir=(1, 0, 0))
    vp_per = Viewport(type=Viewport.Type.Perspective, camera_dir=(-0.56, 0.56, -0.56))

    # Map orientation names to their viewports
    orient_map = {
        'top': vp_top,
        'front': vp_front,
        'left': vp_left,
        'perspective': vp_per
    }

    # Load all dump file data and add it to the scene
    pipeline = import_file(os.path.join(dump_path, 'out*.dump'))
    pipeline.add_to_scene()

    # Render image or animation for each requested view
    for orient in orient_name:
        vp = orient_map.get(orient)
        if not vp:
            print(f"Warning: Unknown orientation '{orient}' — skipping.")
            continue

        vp.zoom_all(size=img_size)

        if pipeline.num_frames == 1:
            vp.render_image(
                filename=os.path.join(gif_path, f'ovito_img_{orient}.png'),
                size=img_size,
                background=(0, 0, 0),
                renderer=TachyonRenderer(ambient_occlusion=False, shadows=False)
            )
        else:
            vp.render_anim(
                filename=os.path.join(gif_path, f'ovito_anim_{orient}.gif'),
                size=img_size,
                fps=10,
                background=(1.0, 1.0, 1.0),
                renderer=TachyonRenderer(ambient_occlusion=False, shadows=False),
                stop_on_error=True
            )

def stitch_gifs(directory: str, gif_fnames: list, sub_dims: tuple, positions: list, output_fname: str = 'stitched_output.gif') -> None:
    '''\
    Function to combine frames of equal-length gifs to form the frames of the output gif
    ie. can create a gif of two gifs playing side-by-side, or 4 gifs in quadrants
     - Adapted from https://stackoverflow.com/questions/51517685/combine-several-gif-horizontally-python
    '''
    
    # Handle Inputs
    if len(gif_fnames) != len(positions): raise Exception("One position must be provided for each file")
    if len(positions) != len(set(positions)): raise Exception("Duplicate positions specified")
    if len(sub_dims) != 2: raise Exception("Invalid dimension specification - must be (width, height)")
    if not all(sub_dim > 0 for sub_dim in sub_dims): raise Exception("Specified dimensions must be positive")
    if not all(isinstance(pos, int) for pos in positions): raise Exception("Positions must be integers")
    if max(positions) >= sub_dims[0]*sub_dims[1]: raise Exception("Positions given are larger than number of sub-images")

    (N_rows, N_cols) = sub_dims

    # Read gifs to stitch
    gifs = [iio.imread(directory + fname) for fname in gif_fnames]

    # Stack in specified positions
    n_frames = min([gif.shape[0] for gif in gifs])
    composite_imgs: list = []
    for i_frame in range(n_frames - 1):
        imgs = [gif[i_frame, :, :, :] for gif in gifs]
        composite_img = np.vstack([np.hstack([imgs[positions[r*N_cols + c]] for c in range(N_cols)]) for r in range(N_rows)])
        composite_imgs.append(composite_img)
    frames = np.stack(composite_imgs, axis=0)

    # Write to new gif and optimize
    iio.imwrite(directory + output_fname, frames)
    optimize(directory + output_fname)