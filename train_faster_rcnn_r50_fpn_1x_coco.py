import os
import glob
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from mim.commands.download import download

def main():
    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='VOC datasets')
    parser.add_argument('--batch', type=int, default=1, help='batch for train')
    args = parser.parse_args()

    voc_datasets = args.datasets
    train_batch = args.batch


    
    category_list = set()
    all_xml = glob.glob(os.path.join(voc_datasets, 'Annotations', '*.xml'))
    for each_xml_file in all_xml:
        tree = ET.parse(each_xml_file)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                for item in child:
                    if item.tag == 'name':
                        category_list.add(item.text)
    category_list = sorted(list(category_list))
    


    os.makedirs('models', exist_ok=True)

    checkpoint_name = 'faster_rcnn_r50_fpn_1x_coco'
    config_fname = checkpoint_name + '.py'

    checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root="models")[0]
    # checkpoint: faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    from mmcv import Config

    cfg = Config.fromfile(os.path.join('models', config_fname))

    ####
    ## modify model configuration file
    ####

    # modify dataset type and path
    #cfg.dataset_type = 'VOCDataset' # default: CocoDataset
    #cfg.data_root = './' # default: data/coco

    cfg.data.train.type = 'VOCDataset'
    cfg.data.train.data_root = './'
    #cfg.data.train.ann_file = 'VOC2012/ImageSets/Main/train.txt'
    cfg.data.train.ann_file = os.path.join(voc_datasets, 'ImageSets/Main/train.txt')
    #cfg.data.train.img_prefix = 'VOC2012/'
    cfg.data.train.img_prefix = voc_datasets

    cfg.data.test.type = 'VOCDataset'
    cfg.data.test.data_root = './'
    cfg.data.test.ann_file = 'VOC2012/ImageSets/Main/val.txt'
    cfg.data.test.img_prefix = 'VOC2012/'

    cfg.data.val.type = 'VOCDataset'
    cfg.data.val.data_root = './'
    cfg.data.val.ann_file = 'VOC2012/ImageSets/Main/val.txt'
    cfg.data.val.img_prefix = 'VOC2012/'

    # number of classes
    cfg.model.roi_head.bbox_head.num_classes = 2 # default: 80

    # checkpoint path
    cfg.load_from = os.path.join('models', checkpoint)

    # learning rate
    cfg.optimizer.lr = 0.02 / 8 # default: 0.02

    # evaluation metric
    cfg.evaluation.metric = 'mAP'

    # modify cuda setting
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    # set seed
    cfg.seed = 0

    # set epochs
    cfg.runner.max_epochs = train_batch #default: 12

    # set output dir
    cfg.work_dir = 'output'

    classes = category_list

    cfg.data.train.classes = category_list
    cfg.data.test.classes = category_list
    cfg.data.val.classes = category_list

    cfg.dump('original_cfg.py')
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    import os.path as osp
    import mmcv

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    model.CLASSES = category_list

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, validate=True)

if __name__ == '__main__':
    main()
