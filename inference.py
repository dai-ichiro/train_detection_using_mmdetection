from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def main():
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
    show_result_pyplot(model, img_path, result, palette='random', score_thr=threshhold)

if __name__ == '__main__':
    main()
