import glob
import os
import time

import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer

from collections import Counter


fontStyle = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 30, encoding="utf-8", index=1)
color = (255, 32, 255)


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    cap = cv2.VideoCapture('parking_lot/13.mp4')
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 50
    vout = cv2.VideoWriter('ssd.avi', fourcc, fps, sz, True)

    count = 0
    # cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            count += 1
            # if count % 3 == 1:
            start = time.time()
            image = frame
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start

            start = time.time()
            result = model(images.to(device))[0]
            inference_time = time.time() - start

            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            obj_dict = Counter(labels)
            scores = scores[indices]
            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print(meters)
            # drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            for i in range(len(labels)):
                if labels[i] == 3:
                    text = 'car:' + str(round(scores[i], 2))
                    cv2.rectangle(image, tuple(boxes[i][:2]), tuple(boxes[i][2:]), color, 3)
                    image = Image.fromarray(image)
                    draw = ImageDraw.Draw(image)
                    draw.text(tuple([boxes[i][0], boxes[i][1]-40]), text, color, font=fontStyle)  
                    image = np.asarray(image)
            cv2.imshow('drawn_image', image)
            vout.write(image)
            if count >= 800 or cv2.waitKey(1) & 0xFF == ord('q'):
                break


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
    parser.add_argument("--dataset_type", default="coco", type=str, help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument("--output_dir", type=str, default='outputs', help="Trained weights.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
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
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
