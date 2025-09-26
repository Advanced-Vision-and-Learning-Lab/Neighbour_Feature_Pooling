
# Neighbour_Feature_Pooling


This repository contains a modular PyTorch Lightning framework for experimenting with Neighbour Feature Pooling (NFP) pooling and various CNN architectures on multiple vision datasets.  
All pipeline results and experiment tracking are handled in `PipelineExperiment.ipynb`, so you can easily save outputs and monitor progress.

---

##  Project Structure
```
Neighbour_Feature_Pooling/
│
├── datasets/
│   ├── DataModules.py         # PyTorch Lightning DataModule definitions for all datasets
│   └── CustomDatasets.py      # Custom dataset classes (PRMI, MSTAR, UCMerced, etc.)
│
├── models/
│   ├── custom_models.py       # SimpleCNN and variants with flexible pooling
│   └── pooling/
│       ├── nsfs.py            # NSFS pooling implementation
│       └── hybrid_pooling.py  # Hybrid pooling layers (e.g., NSFS+MaxPooling)
│
├── lightning_wrappers/
│   └── Lightning_Wrapper.py   # LightningModule wrapper for model training & evaluation
│
├── scripts/                   # (Optional) Training/inference helper scripts
│
├── data/                      # (Ignored by git) Place your datasets here (e.g., UCMerced)
│
├── requirements.txt           # All Python dependencies
├── PipelineExperiment.ipynb   # Main notebook for running, tracking & saving all experiments
├── README.md                  # This file!
└── .gitignore                 # Ensures data/ and other unnecessary files are not tracked
```

---

##  Folders Explained

- **datasets/**  
  Contains Lightning DataModules and custom dataset classes. Add or modify DataModules here for new datasets.

- **models/**  
  Core models and pooling implementations.  
  - `custom_models.py` holds the main CNN architectures.  
  - `pooling/` contains advanced pooling layers (NSFS, hybrid pooling).

- **lightning_wrappers/**  
  LightningModule wrappers for PyTorch Lightning—handles training, validation, logging, etc.

- **scripts/**  
  Optional scripts for CLI-based training or evaluation (not needed for the notebook workflow).

- **data/**  
  Place datasets here. **This folder is ignored by git** (see `.gitignore`).

- **PipelineExperiment.ipynb**  
  The main Jupyter notebook.  
  - Run all experiments here.
  - Save results, visualizations, and logs for reproducibility and easy comparison.

---

##  Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**
   - Download datasets (e.g., UCMerced) to the `data/` folder.

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook PipelineExperiment.ipynb
   ```
   - All pipeline steps (importing modules, model setup, training, evaluation) are organized as notebook cells.
   - Save outputs/results in the notebook for future reference and reproducibility.

---

## Experiment Tracking

- **Why PipelineExperiment.ipynb?**  
  All results, parameters, and code versions are kept in one place for transparency and collaboration.
- Save figures, tables, and logs in notebook cells.
- Comment on results and model changes as you go for easy tracking.

---

##  Data Handling

- The `data/` folder is **not** tracked by git.  
  If you need to ignore additional large files, add them to `.gitignore`.

---

##  Contributions

Feel free to open issues or pull requests for improvements, bugfixes, or new models/pooling layers!

---

##  Project 

- Lab: Advanced Vision and Learning Lab

---

**Happy experimenting!**