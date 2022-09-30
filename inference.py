from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import mmcv

def main():
    from mmdet.utils import register_all_modules
    register_all_modules()
    
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--config', type = str, required=True, help = 'config file')
    parser.add_argument('--checkpoint', type = str, required=True, help = 'checkpoint file')
    parser.add_argument('--threshhold', type = float, default=0.7, help = 'threshhold')
    
    args = parser.parse_args()

    img_path = args.image
    checkpoint = args.checkpoint
    config = args.config
    threshhold = args.threshhold

    model = init_detector(config, checkpoint, device = 'cuda')
    result = inference_detector(model, img_path)
    
    from mmdet.registry import VISUALIZERS
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=True,
            pred_score_thr=threshhold)

if __name__ == '__main__':
    main()