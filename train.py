import argparse
import warnings
import random
import tqdm
# tqdm: 진행상태바를 편하게 보는 라이브러리
import pathlib
# pathlib 모듈: 파일시스템 경로를 단순한 문자열이 아니라 object로 다루자는 것
# 기본(os.path): os.path.join(dir_name, sub_dir_name, file_name)
# Pathlib 방법: dir = Path(dri_name)
# file = dir / sub_dir_name / file_name 식으로 연산자 이용하여 할 수있다.
# 여기서도 Path 선전한 변수에 .glob 이용가능
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils

from model import *
from utils import *

parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary datasets.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1,
                    help='Number of styles to create for each image (default: 1)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1')
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'],
                    help='List of image extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', type=int, default=0,
                    help='If set to anything else than 0, center crop of this size will be applied to the content image \
                    after resizing in order to create a squared image (default: 0)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # distributed 부분 다 삭제

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    # set content and style dir
    content_dir = pathlib.Path(args.content_dir)
    content_dir = content_dir.resolve()

    style_dir = pathlib.Path(args.style_dir)
    style_dir = style_dir.resolve()
    #path.resolve(): 전체경로를 반환 , resolve().parent: 상위 dir

    output_dir = pathlib.Path(args.output_dir)
    output_dir = output_dir.resolve()
    assert style_dir.is_dir() # dir인가 확인

    # content 이미지 가져오기
    extentions = args.extensions # extensions[jpg, png, jpeg]
    dataset = []
    for ext in extentions:
        dataset += list(content_dir.rglob('*.' + ext))
        # rglob은 하위dir까지 다 가져오고, glob은 그 dir에서만 가져옴

    contents = sorted(dataset)
    print("content_image: ", len(contents))

    # style 이미지 가져오기
    styles = []
    for ext in extentions:
        styles += list(style_dir.rglob('*.' + ext))

    styles = sorted(styles)
    print("style_image: ", len(styles))

#todo 코드를 다시 찾아보기 train부분이 없음
#todo https://github.com/naoto0804/pytorch-AdaIN/blob/9076bdac962f0c744389636dc47f8a3351974009/train.py#L22








def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))

    if crop != 0:
        transform_list.append(transforms.CenterCrop(crop))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style

if __name__ == '__main__':
    main()









