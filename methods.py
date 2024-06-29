import warnings
import torch
import cv2

from torch.nn.parallel import DistributedDataParallel as NativeDDP
from data.transforms import Multispectral_Transforms
from engine.prototypical_networks import PrototypicalNetworks
from feature_extractor.dofa import DOFAFeatureExtractor
from utils.constants import SamMethod, MainMethod
from data.fewshot_data import get_batch_prototypes
from sam_proposal import FASTSAM, MobileSAM, SAM, EdgeSAM
from utils import *
from pycocotools.coco import COCO

WAVELENGHTS = {
    "Blue": 0.45,
    "Green": 0.56, 
    "Red": 0.65, 
    "RE": 0.73,
    "NIR": 0.84
}

def get_wavelengths(bands, wavelenghts):
    results = []
    for color in bands:
        if color in wavelenghts:
            results.append(wavelenghts[color])
        elif color == "RGB":
            results.append(wavelenghts["Red"])
            results.append(wavelenghts["Green"])
            results.append(wavelenghts["Blue"])
    return results


def few_shot(args, is_single_class=None, output_root=None, fewshot_method=None, bands=["RGB"]):
    """ Use sam and fewshot to classify the masks.
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    # STEP 1: create data loaders
    # the loaders will load the images in numpy arrays
    labeled_loader, test_loader, _, _, validation_loader = create_datasets_and_loaders(args)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_loader_to_json(test_loader, output_root, filename="test")
    save_loader_to_json(validation_loader, output_root, filename="validation")

    # STEP 2: create an SAM instance
    if args.sam_proposal == SamMethod.MOBILE_SAM:
        sam = MobileSAM(args)
        sam.load_simple_mask()
    elif args.sam_proposal == SamMethod.FAST_SAM:
        sam = FASTSAM(args)
        sam.load_simple_mask()
    else:
        sam = SAM(args)
        sam.load_simple_mask()

    # STEP 3: create few-shot model    
    feature_extractor = DOFAFeatureExtractor(
        checkpoint=None, #args.checkpoint, 
        model_size="base",
        wave_list= get_wavelengths(args.bands.split(","), WAVELENGHTS) #["RGB", "NIR"]
    )

    if fewshot_method == MainMethod.FEWSHOT_1_CLASS:
        fs_model = PrototypicalNetworks(
            is_single_class=is_single_class, 
            use_sam_embeddings=args.use_sam_embeddings,
            backbone=feature_extractor, 
            use_softmax=False,
            device=args.device
        ).to(args.device)

    # STEP 4: get the raw support set
    trans_norm = None
    if args.use_sam_embeddings:
        #trans_norm = Transform_To_Models()
        trans_norm = Multispectral_Transforms(bands_to_apply=args.bands.split(","))
    elif feature_extractor.is_transformer:
        #trans_norm = Transform_To_Models(
        #        size=feature_extractor.input_size, 
        #        force_resize=True, keep_aspect_ratio=False
        #    )
        trans_norm = Multispectral_Transforms(bands_to_apply=args.bands.split(","))
    else:
        #trans_norm = Transform_To_Models(
        #        size=33, force_resize=False, keep_aspect_ratio=True
        #    )
        trans_norm = Multispectral_Transforms(bands_to_apply=args.bands.split(","))

    if is_single_class:
        # single class does not require background class
        imgs, _ = get_batch_prototypes( 
            labeled_loader, args.num_classes,
            get_background_samples=False, # single class
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings
        )
        #  create the prototypes
        fs_model.process_support_set(imgs) # just one class
    else:
        # REQUIRES background class
        imgs, labels = get_batch_prototypes(
            labeled_loader, args.num_classes, 
            get_background_samples=True, # two classes
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings
        )
        # create prototypes
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [i-1 for i in labels]    
        fs_model.process_support_set(imgs, labels)
    
    # STEP 5: classify these inferences using the few-shot model
    # and SAM predictions.
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/test.json"
    coco_gt = COCO(f"{gt_eval_path}")
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results.json"

    if is_single_class:
        save_inferences_singleclass(
            fs_model, test_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings
        )

        save_inferences_singleclass(
            fs_model, validation_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings, val=True
        )
    else:
        save_inferences_twoclasses(
            fs_model, test_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings
        )

        save_inferences_twoclasses(
            fs_model, validation_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings, val=True
        )

    # STEP 6: evaluate model
    if is_single_class:
        # for idx_ in range(1,4):
        idx_ = 2
        MAX_IMAGES = 100000
        gt_eval_path = f"{output_root}/test.json"
        coco_gt = COCO(gt_eval_path)
        image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
        res_data = f"{output_root}/bbox_results_std{idx_}.json"
        eval_sam(
            coco_gt, image_ids, res_data, 
            output_root, method=args.method,
            number=idx_
        )

        # Validation 
        gt_val_path = f"{output_root}/validation.json"
        coco_val_gt = COCO(gt_val_path)
        image_val_ids = coco_val_gt.getImgIds()[:MAX_IMAGES]
        res_val_data = f"{output_root}/bbox_results_val_std{idx_}.json"

        eval_sam(
            coco_val_gt, image_val_ids, res_val_data, 
            output_root, method=args.method,
            number=idx_, val=True
        )

    else:
        eval_sam(
            coco_gt, image_ids, res_data, 
            output_root, method=args.method
        )

        # Validation 
        gt_val_path = f"{output_root}/validation.json"
        coco_val_gt = COCO(gt_val_path)
        image_val_ids = coco_val_gt.getImgIds()[:MAX_IMAGES]
        res_val_data = f"{output_root}/bbox_results_val.json"

        eval_sam(
            coco_val_gt, image_val_ids, res_val_data, 
            output_root, method=args.method,
            val=True
        )

if __name__ == '__main__':
    args = get_parameters()
    root_output = "./output/" #/content/drive/MyDrive/Agro-Pineapples/output/" #"./output/"

    bands = "_".join(args.bands.split(","))

    if not args.numa == -1:
        throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)

    #if args.use_sam_embeddings:
    #    output_root = f"{root_output}{args.output_folder}/seed{args.seed}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}/{args.method}@samEmbed@{args.sam_proposal}"
    #else:
    output_root = f"{root_output}{args.output_folder}/seed{args.seed}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}/{args.method}@{args.timm_model}@{args.sam_proposal}/{bands}"
    if args.method == MainMethod.FEWSHOT_1_CLASS:
        few_shot(args, is_single_class=True, output_root=output_root, fewshot_method=args.method, bands=["RGB"] if args.bands == "RGB" else args.bands.split(","))
    elif args.method == MainMethod.FEWSHOT_2_CLASSES:
        few_shot(args, is_single_class=False, output_root=output_root, fewshot_method=args.method, bands=["RGB"] if args.bands == "RGB" else args.bands.split(","))
