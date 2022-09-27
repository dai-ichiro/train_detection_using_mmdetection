
import argparse
import cv2
from mmdet.apis import inference_detector, init_detector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--threshhold', type=float, default=0.9, help='Bbox score threshold')
    parser.add_argument('--camera_id', type=int, default=0, help='camera device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device = 'cuda')

    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()

        result = inference_detector(model, frame)

        for i in range(len(model.CLASSES)):
            count = sum([True if x[4] > args.threshhold else False for x in result[i]])
            print(f'class {i+1}: {model.CLASSES[i]}   {count}   ', end='')
        
        print()
        print()

        result_img = model.show_result(frame, result, score_thr=args.threshhold)
        cv2.imshow('result', result_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

