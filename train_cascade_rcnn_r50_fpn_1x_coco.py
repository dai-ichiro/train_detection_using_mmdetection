import os
import glob
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from mim.commands.download import download
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='VOC dataset')
    parser.add_argument('--batch', type=int, default=1, help='batch for train')
    parser.add_argument('--lr', type=float, default=0.0025, help='initial learning rate')
    parser.add_argument('--seed', type=int, default=0, help="seed")
    args = parser.parse_args()

    voc_dataset = args.data
    train_batch = args.batch
    learning_rate = args.lr
    seed = args.seed

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
    print('class:')
    print('\n'.join(category_list))

    os.makedirs('models', exist_ok=True)

    checkpoint_name = 'cascade_rcnn_r50_fpn_1x_coco'
    config_fname = checkpoint_name + '.py'

    checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root="models")[0]
    # checkpoint = faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    cfg = Config.fromfile(os.path.join('models', config_fname))

    ####
    ## modify configuration file
    ####

    cfg.data.train.type = 'VOCDataset'
    cfg.data.train.data_root = './'
    cfg.data.train.ann_file = os.path.join(voc_dataset, 'ImageSets/Main/train.txt')
    cfg.data.train.img_prefix = voc_dataset

    cfg.data.test.type = 'VOCDataset'
    cfg.data.test.data_root = './'
    cfg.data.test.ann_file = os.path.join(voc_dataset, 'ImageSets/Main/val.txt')
    cfg.data.test.img_prefix = voc_dataset

    cfg.data.val.type = 'VOCDataset'
    cfg.data.val.data_root = './'
    cfg.data.val.ann_file = os.path.join(voc_dataset, 'ImageSets/Main/val.txt')
    cfg.data.val.img_prefix = voc_dataset

    # number of classes
    for i in range(3):
        cfg.model.roi_head.bbox_head[i].num_classes = len(category_list)

    # checkpoint path
    cfg.load_from = os.path.join('models', checkpoint)

    # learning rate
    cfg.optimizer.lr = learning_rate # default: 0.02

    # evaluation metric
    cfg.evaluation.metric = 'mAP'

    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # set seed
    cfg.seed = seed

    # set epochs
    cfg.runner.max_epochs = train_batch #default: 12

    # set output dir
    cfg.work_dir = 'output'
    os.makedirs(cfg.work_dir, exist_ok=True)

    # set class names
    cfg.data.train.classes = category_list
    cfg.data.test.classes = category_list
    cfg.data.val.classes = category_list

    # save new config
    cfg.dump('finetune_cfg.py')
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    model.CLASSES = category_list

    # train
    train_detector(model, datasets, cfg, validate=True)

if __name__ == '__main__':
    main()
