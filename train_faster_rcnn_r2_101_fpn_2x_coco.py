import os
import sys
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
    parser.add_argument('--data', type=str, required=True, help='path to VOC dataset')
    parser.add_argument('--epochs', type=int, default=3, help='total train epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help="seed")
    return parser.parse_args()

def main(opt):
    voc_dataset = opt.data
    train_epochs = opt.epochs
    learning_rate = opt.lr
    seed = opt.seed
    
    voc_dataset_dir, voc_dataset_name = os.path.split(voc_dataset)

    category_list = set()
    all_xml = glob.glob(os.path.join(voc_dataset, 'Annotations', '*.xml'))
    for each_xml_file in all_xml:
        tree = ET.parse(each_xml_file)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                for item in child:
                    if item.tag == 'name':
                        category_list.add(item.text)
    category_list = sorted(list(category_list))
    
    print(f'dataset: {voc_dataset}')
    print(f'num of classes: {len(category_list)}')
    print(f'class: {category_list}')

    annotationfile_list = glob.glob(os.path.join(voc_dataset, 'ImageSets/Main/*'))
    
    each_files = [os.path.basename(x) for x in annotationfile_list]
    # for train
    if 'train.txt' in each_files:
        ann_file_train = os.path.join(voc_dataset_name, 'ImageSets/Main/train.txt')
    elif 'trainval.txt' in each_files:
        ann_file_train = os.path.join(voc_dataset_name, 'ImageSets/Main/trainval.txt')
    else:
        print('[Error] No such file: train.txt or trainval.txt')
        sys.exit()
    # for test
    if 'test.txt' in each_files:
        ann_file_test = os.path.join(voc_dataset_name, 'ImageSets/Main/test.txt')
    elif 'testval.txt' in each_files:
        ann_file_test = os.path.join(voc_dataset_name, 'ImageSets/Main/testval.txt')
    elif 'valtest.txt' in each_files:
        ann_file_test = os.path.join(voc_dataset_name, 'ImageSets/Main/valtest.txt')
    elif 'val.txt' in each_files:
        ann_file_test = os.path.join(voc_dataset_name, 'ImageSets/Main/val.txt')
    else:
        ann_file_test = ann_file_train
    # for val
    if 'val.txt' in each_files:
        ann_file_val = os.path.join(voc_dataset_name, 'ImageSets/Main/val.txt')
    elif 'testval.txt' in each_files:
        ann_file_val = os.path.join(voc_dataset_name, 'ImageSets/Main/testval.txt')
    elif 'valtest.txt' in each_files:
        ann_file_val = os.path.join(voc_dataset_name, 'ImageSets/Main/valtest.txt')
    elif 'test.txt' in each_files:
        ann_file_val = os.path.join(voc_dataset_name, 'ImageSets/Main/test.txt')
    else:
        ann_file_val = ann_file_train

    print(f'dataset path: {voc_dataset_dir}')
    print(f'annotation file for train: {ann_file_train}')
    print(f'annotation file for val: {ann_file_val}')
    print(f'annotation file for test: {ann_file_test}')
    
    os.makedirs('models', exist_ok=True)

    #checkpoint_name = model_name
    checkpoint_name = 'faster_rcnn_r2_101_fpn_2x_coco'
    config_fname = checkpoint_name + '.py'

    checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root="models")[0]

    cfg = Config.fromfile(os.path.join('models', config_fname))

    ####
    ## modify configuration file
    ####

    cfg.data.train.type = 'VOCDataset'
    cfg.data.test.type = 'VOCDataset'
    cfg.data.val.type = 'VOCDataset'

    cfg.data.train.data_root = voc_dataset_dir
    cfg.data.test.data_root = voc_dataset_dir
    cfg.data.val.data_root = voc_dataset_dir


    cfg.data.train.ann_file = ann_file_train
    cfg.data.test.ann_file = ann_file_test
    cfg.data.val.ann_file = ann_file_val

    cfg.data.train.img_prefix = voc_dataset_name
    cfg.data.test.img_prefix = voc_dataset_name
    cfg.data.val.img_prefix = voc_dataset_name

    # number of classes (default: 80)
    cfg.model.roi_head.bbox_head.num_classes = len(category_list) 

    # checkpoint path
    cfg.load_from = os.path.join('models', checkpoint)

    # learning rate (default: 0.02)
    if learning_rate is not None:
        cfg.optimizer.lr = learning_rate
    else:
        cfg.optimizer.lr = cfg.optimizer.lr / 8
 
    # evaluation metric
    cfg.evaluation.interval = 2
    cfg.evaluation.metric = 'mAP'

    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # set seed
    cfg.seed = seed

    # set epochs (#default: 12)
    cfg.runner.max_epochs = train_epochs 

    # checkpoint_config
    cfg.checkpoint_config.interval = 2
    
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
