# Neighbour Feature Pooling (NFP)  
**Texture-Aware Pooling for Remote Sensing and Plant Phenotyping**  

 *By Fahimeh Orvati Nia, Amirmohammad Mohammadi, Salim Al Kharsa, Pragati Naikare, Zigfried Hampel-Arias, and Joshua Peeples*  

---

##  Overview  
Neighbour Feature Pooling (NFP) is a novel pooling mechanism that captures local neighborhood similarity for improved **texture-aware classification** in remote sensing and agricultural image datasets.  
This repository provides the PyTorch Lightning implementation for training and evaluating **ResNet18, ResNet50, MobileNetV3, and ViT-Tiny backbones** with GAP, NFP, fractal, lacunarity, RADAM, and DeepTEN pooling modules.  

<p align="center">
  <img src="docs/figures/nfp_overview.png" width="600"/>
</p>  

---

##  Datasets Supported  
-  **UCMerced Land Use** (21 classes)  
-  **RESISC45** (45 classes)  
-  **GTOS-Mobile** (31 classes)  
-  **PlantVillage** (15–38 classes)  
-  **EuroSAT** (10 classes, 13 spectral bands)  
-  **MSTAR** (SAR imagery, 6 classes)  
-  **CIFAR-10** (baseline sanity check)  

---

##  Installation  

```bash
# clone the repo
git clone https://github.com/fahimehorvatinia/Neighbour_Feature_Pooling_Clean.git
cd Neighbour_Feature_Pooling_Clean

# create environment
module purge
module load GCC/13.3.0
module load Python/3.12.3
python -m venv myenv
source myenv/bin/activate

# install requirements
pip install -r requirements.txt
```

---

##  Training Demo  

Example: train **ResNet18 (GAP only) on EuroSAT**  

```bash
EXPERIMENT_NAME="gap_only-resnet18-eurosat"
DATASET="EuroSAT"
DATA_DIR="data/EuroSAT"

mkdir -p logs/${EXPERIMENT_NAME}

python demo.py     --name ${EXPERIMENT_NAME}     --dataset ${DATASET}     --data_dir ${DATA_DIR}     --model_type resnet18     --model_variant gap_only
```

Results (logs + checkpoints) are saved in:  
```bash
logs/${DATASET}/${EXPERIMENT_NAME}
checkpoints/${DATASET}/${EXPERIMENT_NAME}
```

---

##  Example Results  

| Model          | Dataset     | Pooling        | Accuracy (%) |
|----------------|-------------|----------------|--------------|
| ResNet18       | UCMerced    | GAP            | 87.1         |
| ResNet18       | UCMerced    | **NFP (cosine)** | **91.5**     |
| MobileNetV3    | GTOS-Mobile | RADAM          | 78.3         |
| ViT-Tiny       | PlantVillage| Lacunarity     | 95.0         |

---

##  Repository Structure  
s
```
Neighbour_Feature_Pooling
│── demo.py                # Main training script
│── models/                # ResNet, MobileNet, ViT variants + pooling modules
│── datasetsnew/           # PyTorch Lightning DataModules
│── lightning_wrappers/    # Lightning Wrapper for training/evaluation
│── Extra_files/           # Environment + configs
│── requirements.txt
│── README.md
```

---

##  License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

---

##  Citation  

If you use this repository in your research, please cite:  

```bibtex
@inproceedings{orvatinia2026nfp,
  title     = {Neighbour Feature Pooling for Remote Sensing Image Classification},
  author    = {Orvati Nia, Fahimeh and Mohammadi, Amirmohammad and Al Kharsa, Salim and Naikare, Pragati and Hampel-Arias, Zigfried and Peeples, Joshua},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```
