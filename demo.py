# -*- coding: utf-8 -*-
"""
Train and evaluate models with GAP/NFP variants on various datasets.
@author: jpeeples, forvatinia
"""
import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import timm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as vutils
from models.pooling.nfp import NFPPooling
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from lightning_wrappers.Lightning_Wrapper import Lightning_Wrapper
from datasetsnew.DataModules import UCMerced_DataModule, RESISC45_DataModule, GTOSMobile_DataModule, PlantVillage_DataModule, EuroSAT_DataModule
import argparse
import sys
import importlib
# from models.resnet18 import (
#     RESNET18_GAP_ONLY,
#     RESNET18_GAP_MLP,
#     RESNET18_NFP_CONV_ONLY,
#     RESNET18_NFP_CONV_MLP,
#     RESNET18_GAP_NFP_CONV_NOMLP_CONCAT,
#     RESNET18_GAP_NFP_NOCONV_NOMLP_CONCAT,
#     RESNET18_GAP_NFP_CONV_MLP_CONCAT,
#     RESNET18_GAP_NFP_NOCONV_MLP_CONCAT,
#     RESNET18_NFP_AT_LAYER,
#     ResNet18_NFPHeadWithSEGate,
# )
# from models.mobilenetv3 import (
#     MOBILENETV3_GAP_ONLY,
#     MOBILENETV3_GAP_MLP,
#     MOBILENETV3_NFP_CONV_ONLY,
#     MOBILENETV3_NFP_CONV_MLP,
#     MOBILENETV3_GAP_NFP_CONV_NOMLP_CONCAT,
#     MOBILENETV3_GAP_NFP_NOCONV_NOMLP_CONCAT,
#     MOBILENETV3_GAP_NFP_CONV_MLP_CONCAT,
#     MOBILENETV3_GAP_NFP_NOCONV_MLP_CONCAT,
#     MOBILENETV3_NFP_INSERT,
# )
# from models.vittiny import (
#     VITTINY_GAP_ONLY,
#     VITTINY_GAP_MLP,
#     VITTINY_NFP_CONV_ONLY,
#     VITTINY_NFP_CONV_MLP,
#     VITTINY_GAP_NFP_CONV_NOMLP_CONCAT,
#     VITTINY_GAP_NFP_NOCONV_NOMLP_CONCAT,
#     VITTINY_GAP_NFP_CONV_MLP_CONCAT,
#     VITTINY_GAP_NFP_NOCONV_MLP_CONCAT,
# )
from models.texture_pooling import (
    ResNet18_FractalPooling, ViTTiny_FractalPooling, MobileNetV3_FractalPooling,
    ResNet18_NFPPooling, ViTTiny_NFPPooling, MobileNetV3_NFPPooling,
    ResNet18_LacunarityPooling, ViTTiny_LacunarityPooling, MobileNetV3_LacunarityPooling,
    ResNet18_DeepTENPooling, ViTTiny_DeepTENPooling, MobileNetV3_DeepTENPooling,
    ResNet18_RADAMPooling, ViTTiny_RADAMPooling, MobileNetV3_RADAMPooling, MobileNetV3_NFPPooling_Intermediate, 
    MobileNetV3_MidNFP, MobileNetV3_MultiStageNFP,
    ResNet50_FractalPooling, ResNet50_NFPPooling, ResNet50_LacunarityPooling, ResNet50_DeepTENPooling, ResNet50_RADAMPooling,
    ResNet50_GAPOnly, RESNET18_GAP_ONLY, MOBILENETV3_GAP_ONLY, VITTINY_GAP_ONLY
)
from torchinfo import summary
import inspect

MODEL_SUMMARY_PRINTED = False

def setup_logging(exp_dir):
    log_file = os.path.join(exp_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def get_datamodule(dataset, config):
    if dataset == 'UCMerced':
        return UCMerced_DataModule(
            resize_size=config['resize_size'],
            input_size=config['input_size'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    elif dataset == 'RESISC45':
        return RESISC45_DataModule(
            resize_size=config['resize_size'],
            input_size=config['input_size'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    elif dataset == 'MSTAR':
        return MSTAR_DataModule(
            resize_size=config['resize_size'],
            input_size=config['input_size'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    elif dataset == 'cifar10':
        return CIFAR10_DataModule(
            resize_size=config['resize_size'],
            input_size=config['input_size'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    elif dataset == 'GTOS-Mobile':
        dm = GTOSMobile_DataModule(
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        dm.setup()
        config['num_classes'] = dm.num_classes
        return dm
    elif dataset == 'PlantVillage':
        dm = PlantVillage_DataModule(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            data_dir=config['data_dir']
        )
        dm.setup()
        config['num_classes'] = dm.num_classes
        return dm
    elif dataset == 'EuroSAT':
        return EuroSAT_DataModule(
            resize_size=config['resize_size'],
            input_size=config['input_size'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_model_class(model_type, model_variant):
    # Map model_type and model_variant to the correct class
    if model_type == 'resnet18':
        mapping = {
            'gap_only': RESNET18_GAP_ONLY,
            # 'gap_mlp': RESNET18_GAP_MLP,
            # 'nfp_conv_only': RESNET18_NFP_CONV_ONLY,
            # 'nfp_conv_mlp': RESNET18_NFP_CONV_MLP,
            # 'gap_nfp_conv_nomlp_concat': RESNET18_GAP_NFP_CONV_NOMLP_CONCAT,
            # 'gap_nfp_noconv_nomlp_concat': RESNET18_GAP_NFP_NOCONV_NOMLP_CONCAT,
            # 'gap_nfp_conv_mlp_concat': RESNET18_GAP_NFP_CONV_MLP_CONCAT,
            # 'gap_nfp_noconv_mlp_concat': RESNET18_GAP_NFP_NOCONV_MLP_CONCAT,
            # 'nfp_at_layer': RESNET18_NFP_AT_LAYER,
            # 'se_gate': ResNet18_NFPHeadWithSEGate,
            'texture_fractal': ResNet18_FractalPooling,
            'texture_nfp': ResNet18_NFPPooling,
            'texture_lacunarity': ResNet18_LacunarityPooling,
            'texture_deepten': ResNet18_DeepTENPooling,
            'texture_radam': ResNet18_RADAMPooling
        }
    elif model_type == 'resnet50':
        mapping = {
            'gap_only': ResNet50_GAPOnly,
            'texture_fractal': ResNet50_FractalPooling,
            'texture_nfp': ResNet50_NFPPooling,
            'texture_lacunarity': ResNet50_LacunarityPooling,
            'texture_deepten': ResNet50_DeepTENPooling,
            'texture_radam': ResNet50_RADAMPooling
        }
    elif model_type == 'mobilenetv3' or model_type == 'mobilenetv3_large_100':
        mapping = {
            'gap_only': MOBILENETV3_GAP_ONLY,
            # 'gap_mlp': MOBILENETV3_GAP_MLP,
            # 'nfp_conv_only': MOBILENETV3_NFP_CONV_ONLY,
            # 'nfp_conv_mlp': MOBILENETV3_NFP_CONV_MLP,
            # 'gap_nfp_conv_nomlp_concat': MOBILENETV3_GAP_NFP_CONV_NOMLP_CONCAT,
            # 'gap_nfp_noconv_nomlp_concat': MOBILENETV3_GAP_NFP_NOCONV_NOMLP_CONCAT,
            # 'gap_nfp_conv_mlp_concat': MOBILENETV3_GAP_NFP_CONV_MLP_CONCAT,
            # 'gap_nfp_noconv_mlp_concat': MOBILENETV3_GAP_NFP_NOCONV_MLP_CONCAT,
            # 'nfp_insert': MOBILENETV3_NFP_INSERT,
            'texture_fractal': MobileNetV3_FractalPooling,
            'texture_nfp': MobileNetV3_NFPPooling,
            'texture_lacunarity': MobileNetV3_LacunarityPooling,
            'texture_deepten': MobileNetV3_DeepTENPooling,
            'texture_radam': MobileNetV3_RADAMPooling,
            'texture_nfp_intermediate': MobileNetV3_NFPPooling_Intermediate,
            'mid_nfp': MobileNetV3_MidNFP,
            'multi_stage_nfp': MobileNetV3_MultiStageNFP
        }
    elif model_type == 'vittiny' or model_type == 'vit_tiny_patch16_224':
        mapping = {
            'gap_only': VITTINY_GAP_ONLY,
            # 'gap_mlp': VITTINY_GAP_MLP,
            # 'nfp_conv_only': VITTINY_NFP_CONV_ONLY,
            # 'nfp_conv_mlp': VITTINY_NFP_CONV_MLP,
            # 'gap_nfp_conv_nomlp_concat': VITTINY_GAP_NFP_CONV_NOMLP_CONCAT,
            # 'gap_nfp_noconv_nomlp_concat': VITTINY_GAP_NFP_NOCONV_NOMLP_CONCAT,
            # 'gap_nfp_conv_mlp_concat': VITTINY_GAP_NFP_CONV_MLP_CONCAT,
            # 'gap_nfp_noconv_mlp_concat': VITTINY_GAP_NFP_NOCONV_MLP_CONCAT,
            'texture_fractal': ViTTiny_FractalPooling,
            'texture_nfp': ViTTiny_NFPPooling,
            'texture_lacunarity': ViTTiny_LacunarityPooling,
            'texture_deepten': ViTTiny_DeepTENPooling,
            'texture_radam': ViTTiny_RADAMPooling
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return mapping[model_variant]

def run_experiment(seed, config):
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare data
    data_module = get_datamodule(config['dataset'], config)
    # Force num_input_channels for EuroSAT
    if config['dataset'].lower() == 'eurosat':
        data_module.num_input_channels = 13
    else:
        data_module.num_input_channels = 3
    num_input_channels = data_module.num_input_channels

    # Setup logging directory
    exp_dir = os.path.join(
        'logs',
        config['dataset'],
        f"{config['model_type']}-{config['model_variant']}-seed{seed}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)
    logger.info(f"Starting experiment with seed {seed}")
    logger.info(f"Config: {config}")

    # Download data if needed before setup
    data_module.prepare_data()
    data_module.setup(stage="fit")
    # Print the first batch shape for debugging (if method exists)
    if hasattr(data_module, 'print_first_batch_shape'):
        data_module.print_first_batch_shape()
    # Now safe to access train_dataloader
    sample = next(iter(data_module.train_dataloader()))

    # Input shape & NFP hyper-params
    input_shape = (num_input_channels, config['input_size'], config['input_size'])
    nfp_kwargs = {
        'R':        config.get('nfp_radius', 1),
        'measure':  config.get('similarity', 'cosine'),
        'padding':  config.get('nfp_padding', 0),
        'stride':   config.get('nfp_stride', 1),
        'input_size': config['input_size']
    }

    # Resolve generic model class
    ModelClass = get_model_class(config['model_type'], config['model_variant'])
    def filter_kwargs(cls, base_kwargs):
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in base_kwargs.items() if k in allowed}

    # Build Params dict for texture-pooling variants
    Params = {
        "Model_name": config['model_type'],
        "num_ftrs": {
            "resnet18": 512,
            "vit_tiny_patch16_224": 192,
            "mobilenetv3_large_100": 960,
            "mobilenetv3": 960,
            "resnet50": 2048,
            "vittiny": 192
        },
        "Dataset": config['dataset'],
        "num_classes": {
            "UCMerced": 21,
            "RESISC45": 45,
            "MSTAR": 6,
            "cifar10": 10,
            "GTOS-Mobile": 31,
            "PlantVillage": 15,
            "EuroSAT": 10
        },
        "input_size": config['input_size']
    }

    # --- Model instantiation logic ---
    if config['model_type'] == 'resnet18' and config['model_variant'] == 'nfp_at_layer':
        model = ModelClass(
            num_classes=config['num_classes'],
            nfp_kwargs=nfp_kwargs,
            bottleneck_dim=None,
            layer_idx=config.get('nfp_layer_idx', 3),
            input_shape=input_shape,
            num_input_channels=num_input_channels
        )

    elif config['model_type'] in ('mobilenetv3', 'mobilenetv3_large_100') \
         and config['model_variant'] == 'texture_nfp_intermediate':
        # special case: intermediate‐layer ablation
        model = MobileNetV3_NFPPooling_Intermediate(
            num_classes=config['num_classes'],
            Params=Params,
            layer_idx=config.get('nfp_intermediate_layer_idx', None),
            num_input_channels=num_input_channels
        )

    elif config['model_type'] in ('mobilenetv3', 'mobilenetv3_large_100') \
         and config['model_variant'] == 'mid_nfp':
        model = MobileNetV3_MidNFP(
            num_classes=config['num_classes'],
            nfp_mid_layer_idx=config.get('nfp_mid_layer_idx', None),
            num_input_channels=num_input_channels
        )
    elif config['model_type'] in ('mobilenetv3', 'mobilenetv3_large_100') \
         and config['model_variant'] == 'nfp_insert':
        model = ModelClass(
            num_classes=config['num_classes'],
            nfp_insert_idx=config.get('nfp_insert_idx', 1),
            nfp_kwargs=nfp_kwargs,
            input_shape=input_shape,
            num_input_channels=num_input_channels
        )

    elif 'nfp' in config['model_variant'] or 'texture' in config['model_variant']:
        # generic texture-pooling variants
        base_kwargs = dict(
            num_classes=config['num_classes'],
            input_shape=input_shape,
            Params=Params,
            num_input_channels=num_input_channels
        )
        filtered = filter_kwargs(ModelClass, base_kwargs)
        model = ModelClass(**filtered)

    else:
        # all other variants
        base_kwargs = dict(
            num_classes=config['num_classes'],
            input_shape=input_shape,
            num_input_channels=num_input_channels
        )
        filtered = filter_kwargs(ModelClass, base_kwargs)
        model = ModelClass(**filtered)

    # Print model summary once
    print(f"[DEBUG] Using model class: {model.__class__.__name__}")
    # print(f"[DEBUG] Model: {model}")
    global MODEL_SUMMARY_PRINTED
    if not MODEL_SUMMARY_PRINTED:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        summary(
            model,
            input_size=(1, num_input_channels, config['input_size'], config['input_size']),
            device=device
        )
        MODEL_SUMMARY_PRINTED = True

    # Add a runtime assertion to check input channels
    x = sample['image'] if isinstance(sample, dict) else sample[0]
    print(f"[DEBUG] Input batch shape: {x.shape}, Model expects {num_input_channels} channels.")
    assert x.shape[1] == num_input_channels, f"Model expects {num_input_channels} input channels, but got {x.shape[1]} from the data. Check your DataModule and model setup."

    # Wrap in PyTorch Lightning
    lightning_model = Lightning_Wrapper(
        model=model,
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        log_dir=exp_dir,
        freeze_nfp=True,
        unfreeze_epoch=5
    )

    # Callbacks and trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            'checkpoints',
            config['dataset'],
            f"{config['name']}_seed{seed}"
        ),
        filename=f'{config["name"]}-seed{seed}-' + '{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max',
        save_last=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        min_delta=config['min_delta'],
        mode='min',
        verbose=False
    )
    logger_tb = TensorBoardLogger(
        f'logs/{config["dataset"]}',
        name=f'{config["name"]}_seed{seed}'
    )
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=False,
        log_every_n_steps=1
    )

    # Train
    trainer.fit(lightning_model, datamodule=data_module)

    # Load best and test
    best_path = checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_path}\n")
    best_model = Lightning_Wrapper.load_from_checkpoint(
        best_path,
        model=model,
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        log_dir=exp_dir
    )
    results = trainer.test(best_model, datamodule=data_module)
    return results[0]['test_acc']


def main():
    parser = argparse.ArgumentParser(description="Train models with GAP/NFP variants on various datasets")
    parser.add_argument('--name', type=str, default='exp', help='Experiment name for logging/checkpoints')
    parser.add_argument('--data_dir', type=str, default='/scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling/data/UCMerced/UCMerced_LandUse', help='Path to the dataset directory')
    parser.add_argument('--max_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument(
        '--similarity',
        type=str,
        default='cosine',
        choices=[
            'norm', 'cosine', 'dot', 'rmse', 'geman', 'attention', 'emd',
            'canberra', 'hellinger', 'chisquared1', 'chisquared2', 'gfc',
            'pearson', 'jeffrey', 'squaredchord', 'smith', 'sharpened_cosine', 'scs'
        ],
        help='Similarity measure to use in NFP'
    )
    parser.add_argument('--dataset', type=str, default='UCMerced', choices=['UCMerced', 'RESISC45', 'MSTAR', 'cifar10', 'GTOS-Mobile', 'PlantVillage', 'EuroSAT'], help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vittiny', 'mobilenetv3', 'vit_tiny_patch16_224', 'mobilenetv3_large_100'], help='Type of model to use')
    parser.add_argument('--model_variant', type=str, default='gap_only', choices=[
        'gap_only',
        #  'gap_mlp', 'nfp_conv_only', 'nfp_conv_mlp',
        # 'gap_nfp_conv_nomlp_concat', 'gap_nfp_noconv_nomlp_concat',
        # 'gap_nfp_conv_mlp_concat', 'gap_nfp_noconv_mlp_concat',
        # 'nfp_insert', 'nfp_at_layer', 'se_gate',
        'texture_fractal', 'texture_nfp', 'texture_lacunarity', 'texture_deepten', 'texture_radam', 'texture_nfp_intermediate', 'mid_nfp', 'multi_stage_nfp'                  
    ],
    help='Model variant to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--resize_size', type=int, default=256, help='Resize size for images')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for images')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Early stopping min delta')
    parser.add_argument('--nfp_radius', type=int, default=1, help='NFP radius')
    parser.add_argument('--nfp_padding', type=int, default=0, help='NFP padding')
    parser.add_argument('--nfp_stride', type=int, default=1, help='NFP stride')
    parser.add_argument('--nfp_layer_idx', type=int, default=3, help='Layer index for NFP in nfp_at_layer variant (resnet18 only, 0=layer1, 1=layer2, 2=layer3, 3=layer4)')
    parser.add_argument('--nfp_insert_idx', type=int, default=1, help='Layer index for NFP insertion in nfp_insert variant (mobilenetv3 only)')
    parser.add_argument('--nfp_intermediate_layer_idx', type=int, default=1, help='Layer index for NFP intermediate in texture_nfp_intermediate variant (mobilenetv3 only)')
    parser.add_argument(
    '--nfp_mid_layer_idx',
    type=int,
    default=1,
    help='Which intermediate layer to tap for mid‐NFP (1–6)'
)
    args = parser.parse_args()

    # Normalize case for dataset and model_type/model_variant
    args.dataset = args.dataset.strip().replace('-', '').replace('_', '').lower()
    dataset_map = {
        'ucmerced': 'UCMerced',
        'resisc45': 'RESISC45',
        'mstar': 'MSTAR',
        'cifar10': 'cifar10',
        'gtosmobile': 'GTOS-Mobile',
        'plantvillage': 'PlantVillage',
        'eurosat': 'EuroSAT'
    }
    args.dataset = dataset_map.get(args.dataset, args.dataset)
    args.model_type = args.model_type.lower()
    args.model_variant = args.model_variant.lower()

    # Set num_classes based on dataset
    dataset_num_classes = {
        'UCMerced': 21,
        'RESISC45': 45,
        'MSTAR': 6,
        'cifar10': 10,
        'GTOS-Mobile': 31,
        'PlantVillage': 38,
        'EuroSAT': 10
    }
    num_classes = dataset_num_classes[args.dataset]

    # Set data_dir based on dataset
    if args.dataset == 'UCMerced':
        data_dir = '/scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling/data/UCMerced/UCMerced_LandUse'
    elif args.dataset == 'RESISC45':
        data_dir = 'data/RESISC45'
    elif args.dataset == 'GTOS-Mobile':
        data_dir = 'data/GTOS-Mobile'
    elif args.dataset == 'PlantVillage':
        data_dir = 'data/PlantVillage'
    elif args.dataset == 'EuroSAT':
        data_dir = 'data/EuroSAT'
    else:
        data_dir = None
    
    config = {
        'name': args.name,
        'data_dir': args.data_dir,
        'batch_size': {'train': args.batch_size, 'val': args.batch_size, 'test': args.batch_size},
        'num_workers': 12,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'resize_size': args.resize_size,
        'input_size': args.input_size,
        'num_classes': num_classes,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'similarity': args.similarity,
        'dataset': args.dataset,
        'model_type': args.model_type,
        'model_variant': args.model_variant,
        'nfp_radius': args.nfp_radius,
        'nfp_padding': args.nfp_padding,
        'nfp_stride': args.nfp_stride,  
        'nfp_layer_idx': args.nfp_layer_idx,
        'nfp_insert_idx': args.nfp_insert_idx,
        'nfp_intermediate_layer_idx': args.nfp_intermediate_layer_idx,
        'nfp_mid_layer_idx': args.nfp_mid_layer_idx,

    }

    seeds = [42, 123, 999]
    results = []
    for seed in seeds:
        print(f"\n==== Running experiment with seed {seed} ====")
        acc = run_experiment(seed, config)
        print(f"Seed {seed} Test Accuracy: {acc:.4f}")
        results.append(acc)
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print(f"\n Final Test Accuracy over 3 seeds: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == '__main__':
    main()
