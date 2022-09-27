import os
import os.path as osp
import torch
from mim.commands.download import download
from argparse import ArgumentParser

import cv2
import mmcv

from mmtrack.apis import inference_sot, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', help='input video file')
    parser.add_argument(
        '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    parser.add_argument(
        '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    args = parser.parse_args()

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)
    #checkpoint_name = 'siamese_rpn_r50_20e_lasot'
    checkpoint_name = 'stark_st2_r50_50e_lasot'
    checkpoint = download(package='mmtrack', configs=[checkpoint_name], dest_root="models")[0]
    model = init_model(os.path.join('models', checkpoint_name + '.py'), os.path.join('models', checkpoint), device=device)
    
    # load videos
    imgs = mmcv.VideoReader(args.input)

    fps = int(imgs.fps)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.input, img)
            img = mmcv.imread(img_path)
        if i == 0:
            init_bbox = list(cv2.selectROI(args.input, img, False, False))

            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]

        result = inference_sot(model, img, init_bbox, frame_id=i)
        
        model.show_result(
            img,
            result,
            show = True,
            wait_time=int(1000. / fps) if fps else 0,
            out_file = None,
            thickness=args.thickness)
        prog_bar.update()

if __name__ == '__main__':
    main()