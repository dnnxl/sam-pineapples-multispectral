""" Detection dataset
Hacked together by Ross Wightman
"""
import os
import cv2
import torch.utils.data as data
import numpy as np

from pathlib import Path
from PIL import Image
from .parsers import create_parser
from .utils import is_vegetation_index, get_band_combination

class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, data_dir, bands_to_apply=["RGB"], parser=None, parser_kwargs=None, transform=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform
        self.bands_to_apply = bands_to_apply

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(
            img_idx=index, 
            img_size=(img_info['width'], img_info['height']), 
            img_orig_id=img_info['id'],
            img_filename=img_info['file_name']
        )
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        # Multispectral image combination
        im_stacked = None

        if self.bands_to_apply is not None:
            # stack the channels
            file_rgb = str(self.data_dir / img_info['file_name'])
            if self.bands_to_apply and self.bands_to_apply != ["RGB"]:
                bands = []
                root_dir = os.path.dirname(file_rgb)
                imgName = Path(file_rgb).stem
                for band_name in self.bands_to_apply:
                    if band_name == 'RGB' or band_name == 'RGB'.lower():
                        im_rgb = cv2.cvtColor(cv2.imread(file_rgb), cv2.COLOR_BGR2RGB)
                        bands.append(im_rgb)
                    elif is_vegetation_index(band_name):
                        ms_image = []
                        for band in ['Red', 'Green', 'Blue', 'RE', 'NIR']:
                            ms_image.append(cv2.imread(os.path.join(root_dir, f'{imgName}_{band}.TIF'), cv2.IMREAD_GRAYSCALE))
                        ms_image = np.dstack(ms_image)
                        im_vi = get_band_combination(ms_image,band_name)
                        bands.append(im_vi)
                    else:
                        bands.append(cv2.imread(os.path.join(root_dir, f'{imgName}_{band_name}.TIF'), cv2.IMREAD_GRAYSCALE))
                im_stacked = np.dstack(bands)
            else:
                im_stacked = cv2.imread(file_rgb) # BGR, width, height
    
        # Image for SAM model in RGB and Pil format
        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, im_stacked # RGB, _, BGR width height
    
    def get_channels(self):
        if "RGB" in self.bands_to_apply:
            return len(self.bands_to_apply) + 2
        return len(self.bands_to_apply)

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t