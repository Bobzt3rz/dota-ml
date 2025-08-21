# Dota 2 Winrate Prediction

Machine learning project to predict match winrates based on hero compositions using OpenDota public matches data.

## Dataset
- 500k+ recent Dota 2 matches from OpenDota API
- Features: hero compositions, game metadata, skill brackets
- Target: binary classification (Radiant win/loss)

## Setup

1. **Clone and setup environment:**
```bash
git clone <your-repo-url>
cd dota-ml
conda env create -f environment.yml
conda activate dota-ml
```

2. **Add your data:**
   - Place `public_matches_500k.json` in `./data/` directory

3. **Explore the dataset:**
```bash
python explore_dataset.py
```

## Project Structure
```
├── data/                   # Dataset files (not in git)
├── src/                    # Source code
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained models (not in git)
├── results/                # Analysis outputs (not in git)
├── tools/                  # Useful tools
├── environment.yml         # Conda environment
└── requirements.txt        # Pip requirements
```

## Features
- Hero composition (10 heroes per match)
- Game duration, mode, lobby type
- Average skill bracket
- Match outcome prediction

## Models to Try
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Neural Networks

## Results
- Model performance metrics
- Feature importance analysis
- Hero synergy insights
