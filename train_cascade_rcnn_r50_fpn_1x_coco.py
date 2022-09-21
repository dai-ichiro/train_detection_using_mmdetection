import os
import glob
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from mim.commands.download import download
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='folder nama of VOC dataset')
    parser.add_argument('--data_dir', type=str, help='path to folder of VOC dataset')
    parser.add_argument('--epochs', type=int, default=3, help='total train epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help="seed")
    return parser.parse_args()

def main(opt):
    voc_dataset = opt.data
    voc_dataset_dir = opt.data_dir
    train_epochs = opt.epochs
    learning_rate = opt.lr
    seed = opt.seed

    category_list = set()
    all_xml = glob.glob(os.path.join(voc_dataset_dir, voc_dataset, 'Annotations', '*.xml'))
    for each_xml_file in all_xml:
        tree = ET.parse(each_xml_file)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                for item in child:
                    if item.tag == 'name':
                        category_list.add(item.text)
    category_list = sorted(list(category_list))
    
    print(f'dataset: {os.path.join(voc_dataset_dir, voc_dataset)}')
    print(f'num of classes: {len(category_list)}')
    print(f'class: {category_list}')

    annotationfile_list = glob.glob(os.path.join(voc_dataset_dir, voc_dataset, 'ImageSets/Main/*'))
    if len(annotationfile_list) == 1:
        ann_file_train = os.path.join(voc_dataset, 'ImageSets/Main', os.path.basename(annotationfile_list[0]))
        ann_file_test = os.path.join(voc_dataset, 'ImageSets/Main', os.path.basename(annotationfile_list[0]))
        ann_file_val = os.path.join(voc_dataset, 'ImageSets/Main', os.path.basename(annotationfile_list[0]))
    else:
        each_files = [os.path.basename(x) for x in annotationfile_list]
        # for train
        try:
            if 'train.txt' in each_files:
                ann_file_train = os.path.join(voc_dataset, 'ImageSets/Main/train.txt')
            elif 'trainval.txt' in each_files:
                ann_file_train = os.path.join(voc_dataset, 'ImageSets/Main/trainval.txt')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print('[Error] No such file: train.txt or trainval.txt')
        # for test
        try:
            if 'test.txt' in each_files:
                ann_file_test = os.path.join(voc_dataset, 'ImageSets/Main/test.txt')
            elif 'val.txt' in each_files:
                ann_file_test = os.path.join(voc_dataset, 'ImageSets/Main/val.txt')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print('[Error] No such file: test.txt or val.txt')
        # for val
        try:
            if 'val.txt' in each_files:
                ann_file_val = os.path.join(voc_dataset, 'ImageSets/Main/val.txt')
            elif 'test.txt' in each_files:
                ann_file_val = os.path.join(voc_dataset, 'ImageSets/Main/test.txt')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print('[Error] No such file: val.txt or test.txt')
    
    print(f'annotation file for train: {ann_file_train}')
    print(f'annotation file for val: {ann_file_val}')
    print(f'annotation file for test: {ann_file_test}')

    os.makedirs('models', exist_ok=True)

    #checkpoint_name = model_name
    checkpoint_name = 'cascade_rcnn_r50_fpn_1x_coco'
    config_fname = checkpoint_name + '.py'

    checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root="models")[0]

    cfg = Config.fromfile(os.path.join('models', config_fname))

    ####
    ## modify configuration file
    ####

    cfg.data.train.type = 'VOCDataset'
    cfg.data.test.type = 'VOCDataset'
    cfg.data.val.type = 'VOCDataset'

    if voc_dataset_dir is not None:
        cfg.data.train.data_root = voc_dataset_dir
        cfg.data.test.data_root = voc_dataset_dir
        cfg.data.val.data_root = voc_dataset_dir
    else:
        cfg.data.train.data_root = './'
        cfg.data.test.data_root = './'
        cfg.data.val.data_root = './'

    cfg.data.train.ann_file = ann_file_train
    cfg.data.test.ann_file = ann_file_test
    cfg.data.val.ann_file = ann_file_val

    cfg.data.train.img_prefix = voc_dataset
    cfg.data.test.img_prefix = voc_dataset
    cfg.data.val.img_prefix = voc_dataset

    # number of classes (default: 80)
    for i in range(3):
        cfg.model.roi_head.bbox_head[i].num_classes = len(category_list)

    # checkpoint path
    cfg.load_from = os.path.join('models', checkpoint)

    # learning rate (default: 0.02)
    if learning_rate is not None:
        cfg.optimizer.lr = learning_rate
    else:
        cfg.optimizer.lr = cfg.optimizer.lr / 8
 
    # evaluation metric
    cfg.evaluation.metric = 'mAP'

    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # set seed
    cfg.seed = seed

    # set epochs (#default: 12)
    cfg.runner.max_epochs = train_epochs 

    # set output dir
    cfg.work_dir = 'output'
    os.makedirs(cfg.work_dir, exist_ok=True)

    # set class names
    cfg.data.train.classes = category_list
    cfg.data.test.classes = category_list
    cfg.data.val.classes = category_list

    # save new config
    cfg.dump('finetune_cfg.py')
    
    # build dataset
    datasets = [build_dataset(cfg.data.train)]

    # build the detector
    model = build_detector(cfg.model)
    model.CLASSES = category_list

    # train
    train_detector(model, datasets, cfg, validate=True)
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
