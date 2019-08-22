import glob
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
# from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer

fontStyle = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 30, encoding="utf-8", index=1)
label_name = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',11:'fire hydrant',12:'stop sign',13:'parking meter',14:'bench',\
        15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'backpack',26:'umbrella',27:'handbag',28:'tie',29:'suitcase',30:'frisbee',31:'skis',\
        32:'snowboard',33:'sports ball',34:'kite',35:'baseball bat',36:'baseball glove',37:'sports ball',38:'surfboard',39:'tennis racket',40:'bottle',41:'wine glass',42:'cup',43:'fork',44:'knife',\
        45:'spoon',46:'bowl',47:'banana',48:'apple',49:'sandwich',50:'orange',51:'broccoli',52:'carrot',53:'hot dog',54:'pizza',55:'donut',56:'cake',57:'chair',58:'couch',59:'potted plant',60:'bed',\
        61:'dining table',62:'toilet',63:'tv',64:'laptop',65:'mouse',66:'remote',67:'keyboard',68:'cell phone',69:'microwave',70:'oven',71:'toaster',72:'sink',73:'refrigerator',74:'book',75:'clock',\
        76:'vase',77:'scissors',78:'teddy bear',79:'hair drier',80:'toothbrush'} 
color = (0, 255, 0)

@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    model = build_detection_model(cfg)
    model = model.to(cpu_device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(cpu_device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))
        for i in range(len(labels)):
            text = str(label_name[labels[i]]) + str(round(scores[i], 2))
            cv2.rectangle(image, tuple(boxes[i][:2]), tuple(boxes[i][2:]), color, 3)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text(tuple([boxes[i][0], boxes[i][1]-40]), text, color, font=fontStyle)  
            image = np.asarray(image)
        cv2.imshow('drawn_image', image)
        # drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo/test', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
