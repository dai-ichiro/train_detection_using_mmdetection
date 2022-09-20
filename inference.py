from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--config', type = str, required=True, help = 'config file')
    parser.add_argument('--checkpoint', type = str, required=True, help = 'checkpoint file')
    args = parser.parse_args()

    img_path = args.image
    checkpoint = args.checkpoint
    config = args.config

    model = init_detector(config, checkpoint, device = 'cuda')
    result = inference_detector(model, img_path)
    show_result_pyplot(model, img_path, result, score_thr = 0.7)

if __name__ == '__main__':
    main()
