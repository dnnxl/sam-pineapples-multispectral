from pathlib import Path

# Import the function to be tested
from data import create_dataset

if __name__ == '__main__':
    datasets = create_dataset('coco2017', 
                              root="D:/RESEARCH/pineapple_sam/multispectral_gira_10_13_mar21_lote71_5m/", 
                              splits=('train'), bands_to_apply=["RGB", "NIR"])
    for data in datasets:
        img, target, im_stacked = data
        print(img.size)
        print(target["img_filename"])
        print(im_stacked.shape)
        print()
    # get the waveleghts
    