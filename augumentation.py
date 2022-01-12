from typing import Callable, Optional
from albumentations.augmentations.bbox_utils import BboxProcessor
from albumentations.augmentations.geometric.functional import safe_rotate
from albumentations.augmentations.geometric.rotate import RandomRotate90
from dataset import Frame_Box_Dataset

import albumentations
import albumentations.pytorch
import cv2

import draw as d
import numpy as np

class Albumentations:
    def __init__(self, dataset: Frame_Box_Dataset, transform: Optional[Callable] = None, bbox_transform: Optional[Callable] = None, category_ids:Optional[Callable] = [0]):
        self.dataset = dataset
        self.bounding_box = dataset.bounding_box
        self.transform = transform
        self.bbox_transform = bbox_transform
        self.category_ids = category_ids

    def aug_bbox(self, image, boxes, bbox_transform, category_ids):
        augmented_bboxes = bbox_transform(image=image, bboxes=boxes, category_ids=category_ids)
        image = augmented_bboxes['image']
        boxes = augmented_bboxes['bboxes']
        return image, boxes

    def aug(self, image, transform):
        augmented = transform(image=image)
        image = augmented['image']
        return image

    def __call__(self, index: int):
        seq_label,frame_idx,path,idx_bboxes = self.dataset.dataset.iloc[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []
        for i in idx_bboxes:
            if not np.isnan(i):
                bbox = self.bounding_box.iloc[int(i)][['x_min','y_min','x_max','y_max']]
                boxes.append(bbox.values.tolist())
                
        if len(boxes) > 0:
            image, boxes = self.aug_bbox(image, boxes, self.bbox_transform, self.category_ids)
        else:
            image = self.aug(image, self.transform)

        return seq_label,frame_idx, image, boxes

if __name__ == "__main__":
    caminho_frames = 'E:/Projeto/Unifall/Datasets_Ready/imagens'
    caminho_bounding_boxes = 'E:/Projeto/Unifall/Datasets_Ready/imagens/ground_truth.csv'
    save_path = 'E:/Projeto/Unifall/Datasets_Ready/imagens/augumented/'
    category_ids = [0]
    category_id_to_name = {0: 'obj'}

    albumentations_transform_bbox = albumentations.Compose([
    #P é a probabildiade da transformação acontecer, mudar para valor desejado 
    albumentations.HorizontalFlip(p=0.2),
    albumentations.Rotate(limit=90, p=0.2),
    albumentations.VerticalFlip(p=0.2),
    albumentations.RandomRotate90(p=0.2),
    #saferotate rotaciona o frame de entrada pelo range do angulo selecionado definido pelo "limit", por meio de uma distribuição uniforme.
    #border_mode -> flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101
    #albumentations.SafeRotate(limit=90,p=0.2),
    #OneOf garante que um deles vai acontecer caso P seja 1
    albumentations.OneOf([
        albumentations.MotionBlur(p=0.5),
        albumentations.GaussNoise(p=0.5)                 
    ], p=1),
    ], albumentations.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    
    albumentations_transform = albumentations.Compose([
        albumentations.OneOf([
        albumentations.HorizontalFlip(p=0),
        albumentations.Rotate(limit=90, p=0),
        albumentations.VerticalFlip(p=0),
        albumentations.RandomRotate90(p=0),  
        albumentations.SafeRotate(limit=90,p=1),   
    ], p=1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=0.5),
        albumentations.GaussNoise(p=0.5)                 
    ], p=1),
    ])

    dataset = Frame_Box_Dataset(root=caminho_frames,root_bbox=caminho_bounding_boxes)
    aug_dataset = Albumentations(dataset=dataset, transform=albumentations_transform, bbox_transform=albumentations_transform_bbox, category_ids=category_ids)

    for i in range(len(dataset.dataset)):
        seq_label,frame_idx, frame, boxes = aug_dataset(i)
        name = str(seq_label) + '_' + str(i) + '_augmented'
        d.save_img(frame, boxes, save_path, name)

    print(f'comprimento inicial do dataset= {len(dataset)}')       
