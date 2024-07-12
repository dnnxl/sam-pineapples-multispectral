import warnings
import torch
import cv2

from torch.nn.parallel import DistributedDataParallel as NativeDDP
from data.transforms import Multispectral_Transforms
from engine.mahalanobis_filter import MahalanobisFilter
from engine.prototypical_networks import PrototypicalNetworks
from feature_extractor.dofa import DOFAFeatureExtractor
from feature_extractor.timm_models import MyFeatureExtractor
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

    #for images, labels, additional in test_loader:
        # Here images is a batch of images and labels is the corresponding labels
        #print(images.shape)  # Output: torch.Size([64, 1, 28, 28]) if batch_size=64
    #    print(additional)
    #    print("+++++++++++++++++++++++++")
    multispectral = False
    if args.feature_extractor == FeatureExtractor.DOFA_MODEL:
        multispectral = True 
    else:
        multispectral = False

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
    # STEP 3: create few-shot model    
    if args.feature_extractor == FeatureExtractor.DOFA_MODEL:
        if args.dofa_model == DOFAModel.NONE:
            checkpoint = False 
        else:
            checkpoint = True 

        feature_extractor = DOFAFeatureExtractor(
            checkpoint=checkpoint,
            model_size=args.dofa_model,
            wave_list= get_wavelengths(args.bands.split(","), WAVELENGHTS), device=args.device #["RGB", "NIR"]
        )
    else:
        feature_extractor = MyFeatureExtractor(
            args.timm_model, args.load_pretrained, args.num_classes
        )

    if fewshot_method == MainMethod.FEWSHOT_1_CLASS:
        fs_model = PrototypicalNetworks(
            is_single_class=is_single_class, 
            use_sam_embeddings=args.use_sam_embeddings,
            backbone=feature_extractor, 
            use_softmax=False,
            device=args.device,
            multispectral=multispectral
        ).to(args.device)

    # STEP 4: get the raw support set
    trans_norm = None
    if args.feature_extractor == FeatureExtractor.DOFA_MODEL:
        #trans_norm = Transform_To_Models()
        trans_norm = Multispectral_Transforms(size=(224, 224), bands_to_apply=args.bands.split(","))
    elif feature_extractor.is_transformer:
        #trans_norm = Transform_To_Models(
        #        size=feature_extractor.input_size, 
        #        force_resize=True, keep_aspect_ratio=False
        #    )
        trans_norm = Multispectral_Transforms(size=feature_extractor.input_size, bands_to_apply=args.bands.split(","))
    else:
        #trans_norm = Transform_To_Models(
        #        size=33, force_resize=False, keep_aspect_ratio=True
        #    )
        trans_norm = Multispectral_Transforms(size=feature_extractor.input_size, bands_to_apply=args.bands.split(","))



    if is_single_class:
        # single class does not require background class
        imgs, _ = get_batch_prototypes( 
            labeled_loader, args.num_classes,
            get_background_samples=False, # single class
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings, multispectral=multispectral
        )
        #  create the prototypes
        fs_model.process_support_set(imgs) # just one class
    else:
        # REQUIRES background class
        imgs, labels = get_batch_prototypes(
            labeled_loader, args.num_classes, 
            get_background_samples=True, # two classes
            trans_norm=trans_norm,
            use_sam_embeddings=args.use_sam_embeddings, multispectral=multispectral
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
            args.use_sam_embeddings, multispectral=multispectral
        )

        save_inferences_singleclass(
            fs_model, validation_loader, sam, 
            output_root, trans_norm,
            args.use_sam_embeddings, val=True, multispectral=multispectral
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

def mahalanobis_filter(args, is_single_class=True, output_root=None, dim_red="svd", n_components=10, mahalanobis_method="regularization", beta=1):
    """ Use sam and fewshot (maximum likelihood) to classify masks.
    Params
    :args -> parameters from bash.
    :output_root (str) -> output folder location.
    """
    multispectral = False
    if args.feature_extractor == FeatureExtractor.DOFA_MODEL:
        multispectral = True 
    else:
        multispectral = False

    # STEP 1: create data loaders
    labeled_loader, test_loader,_,_, validation_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_loader_to_json(test_loader, output_root, "test")
    save_loader_to_json(validation_loader, output_root, "validation")

    # sam instance - default values of the model
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
    use_dofa_model = False
    if args.feature_extractor == FeatureExtractor.DOFA_MODEL:
        if args.dofa_model == DOFAModel.NONE:
            checkpoint = False 
        else:
            checkpoint = True 

        feature_extractor = DOFAFeatureExtractor(
            checkpoint=checkpoint,
            model_size=args.dofa_model,
            wave_list= get_wavelengths(args.bands.split(","), WAVELENGHTS), device=args.device #["RGB", "NIR"]
        )
        use_dofa_model = True
    else:
        feature_extractor = MyFeatureExtractor(
            args.timm_model, args.load_pretrained, args.num_classes
        )
    #else:


    # instance the main class and instance the timm model
    mahalanobis_filter = MahalanobisFilter(
        sam_model=sam,
        use_sam_embeddings=args.use_sam_embeddings,
        is_single_class=is_single_class,
        dim_red=args.dim_red,
        n_components=args.n_components,
        feature_extractor=feature_extractor,
        bands_to_apply=args.bands,
        multispectral = multispectral, use_dofa_model=use_dofa_model
    )

    # run filter using the backbone, sam, and ood
    mahalanobis_filter.run_filter(
        labeled_loader, test_loader, validation_loader,
        dir_filtered_root=output_root,
        mahalanobis_method=mahalanobis_method, beta=beta, seed=args.seed,
        lambda_mahalanobis=args.mahalanobis_lambda, multispectral=multispectral
    )
    
    # STEP 3: evaluate results
    if is_single_class:
        MAX_IMAGES = 100000
        gt_eval_path = f"{output_root}/test.json"
        coco_gt = COCO(gt_eval_path)
        image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
        res_data = f"{output_root}/bbox_results.json"

        eval_sam(
            coco_gt, image_ids, res_data, 
            output_root, method=args.method,
        )

        gt_eval_path = f"{output_root}/validation.json"
        coco_eval_gt = COCO(gt_eval_path)
        image_eval_ids = coco_eval_gt.getImgIds()[:MAX_IMAGES]
        res_eval_data = f"{output_root}/bbox_results_val.json"

        eval_sam(
            coco_eval_gt, image_eval_ids, res_eval_data, 
            output_root, method=args.method, val=True
        )
    else:
        print("No implemented for multiple class mahalanobis!")
    
        
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
    output_root = f"{root_output}{args.output_folder}/seed{args.seed}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}/{args.method}@{args.feature_extractor}@{args.timm_model}@{args.sam_proposal}/{bands}"
    if args.method == MainMethod.FEWSHOT_1_CLASS:
        few_shot(args, is_single_class=True, output_root=output_root, fewshot_method=args.method, bands=["RGB"] if args.bands == "RGB" else args.bands.split(","))
    elif args.method == MainMethod.FEWSHOT_2_CLASSES:
        few_shot(args, is_single_class=False, output_root=output_root, fewshot_method=args.method, bands=["RGB"] if args.bands == "RGB" else args.bands.split(","))
    elif args.method == MainMethod.FEWSHOT_MAHALANOBIS:
        output_root = f"{root_output}{args.output_folder}/seed{args.seed}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}/{args.method}_{args.mahalanobis}_beta_{args.beta}_lambda_{args.mahalanobis_lambda}@{args.timm_model}@{args.sam_proposal}@{args.dim_red}_{args.n_components}/{bands}"
        mahalanobis_filter(args, is_single_class=True, output_root=output_root, mahalanobis_method=args.mahalanobis, beta=args.beta)