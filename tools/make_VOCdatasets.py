import os
import glob
import random
import cv2
import torch
import mmcv
from mmtrack.apis import inference_sot, init_model
from mim.commands.download import download
import xml.etree.ElementTree as ET

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--videos_dir', type=str, default='videos', help='video folder name' )
parser.add_argument('--split_ratio', type=float, default=0.1, help='validation data / all data')
args = parser.parse_args()

videos_dir = args.videos_dir
split_ratio = args.split_ratio

out_path = 'VOC2012'
annotation_dir = os.path.join(out_path, 'Annotations')
main_dir =  os.path.join(out_path, 'ImageSets/Main')
jpegimages_dir = os.path.join(out_path, 'JPEGImages')

os.makedirs(annotation_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)
os.makedirs(jpegimages_dir, exist_ok=True)

def tracking():
    class_list = glob.glob(os.path.join(videos_dir, '*'))

    class_num = len(class_list)
    print(f'class count = {class_num}')

    video_list = []
    classname_list = []

    for i, each_class in enumerate(class_list):
        if os.path.isdir(each_class):
            classname_without_ext = os.path.basename(each_class)
            classname_list.append(classname_without_ext)
            video_list.append(glob.glob(os.path.join(each_class, '*')))
        else:
            classname_without_ext = os.path.splitext(os.path.basename(each_class))[0]
            classname_list.append(classname_without_ext)
            video_list.append([each_class])

    for i, classname in enumerate(classname_list):
        print(f'class {i}: {classname}')

    for i, videos_in_each_class in enumerate(video_list):
        videos_str = ', '.join(videos_in_each_class)
        print(f'videos of class {i}: {videos_str}')
        
    init_rect_list = []

    for videos_in_each_class in video_list:
        temporary_list_each_class = []
        for video in videos_in_each_class:
            cap = cv2.VideoCapture(video)
            _, img = cap.read()
            cap.release()

            source_window = "draw_rectangle"
            cv2.namedWindow(source_window)
            rect = cv2.selectROI(source_window, img, False, False)
            # rect:(x1, y1, w, h)
            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            rect_convert = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
            temporary_list_each_class.append(rect_convert)
            cv2.destroyAllWindows()
        init_rect_list.append(temporary_list_each_class)

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)
    checkpoint_name = 'siamese_rpn_r50_20e_lasot'
    checkpoint = download(package='mmtrack', configs=[checkpoint_name], dest_root="models")[0]
    model = init_model(os.path.join('models', checkpoint_name + '.py'), os.path.join('models', checkpoint), device=device)

    # tracking
    jpeg_filenames_list = []
    for class_index, videos in enumerate(video_list):
        for video_index, video in enumerate(videos):
            # read video
            frames = mmcv.VideoReader(video)
            h = frames.height
            w = frames.width
            # tracking
            for frame_index, frame in enumerate(frames):
                result = inference_sot(model, frame, init_rect_list[class_index][video_index], frame_id=frame_index)
                bbox = result['track_bboxes']
                # bbox:(x1, y1, x2, y2)
                #center_x = ((bbox[0] + bbox[2]) / 2) / w
                #center_y = ((bbox[1] + bbox[3]) / 2) / h
                #width = (bbox[2] - bbox[0]) / w
                #height = (bbox[3] - bbox[1]) /h

                filename = '%d_%d_%06d'%(class_index, video_index, frame_index)

                # save image
                jpeg_filename = filename + '.jpg'
                cv2.imwrite(os.path.join(jpegimages_dir, jpeg_filename), frame)

                # make image file list
                jpeg_filenames_list.append(filename)

                # save XML file
                xml_filename = filename + '.xml'

                new_root = ET.Element('annotation')

                ET.SubElement(new_root, 'filename').text = jpeg_filename
                
                Size = ET.SubElement(new_root, 'size')
                ET.SubElement(Size, 'width').text = str(w)
                ET.SubElement(Size, 'height').text = str(h)
                ET.SubElement(Size, 'depth').text = '3'

                Object = ET.SubElement(new_root, 'object')
                ET.SubElement(Object, 'name').text = classname_list[class_index]
                ET.SubElement(Object, 'difficult').text = '0'

                Bndbox = ET.SubElement(Object, 'bndbox')
                ET.SubElement(Bndbox, 'xmin').text = str(int(bbox[0]))
                ET.SubElement(Bndbox, 'ymin').text = str(int(bbox[1]))
                ET.SubElement(Bndbox, 'xmax').text = str(int(bbox[2]))
                ET.SubElement(Bndbox, 'ymax').text = str(int(bbox[3]))

                new_tree = ET.ElementTree(new_root) 

                new_tree.write(os.path.join(annotation_dir, xml_filename))

    # save text file
    num_val = int(len(jpeg_filenames_list) * split_ratio)
    val_data_list = random.sample(jpeg_filenames_list, num_val)
    train_data_list = list(set(jpeg_filenames_list) - set(val_data_list))

    with open(os.path.join(main_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_data_list))
    
    with open(os.path.join(main_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_data_list))

if __name__ == '__main__':
    tracking()

