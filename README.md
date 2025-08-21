# Dota 2 Winrate Prediction

Machine learning project to predict match winrates based on hero compositions using OpenDota public matches data.

## Dataset
- **500k+ recent Dota 2 matches** from OpenDota API (Aug 14-20, 2025)
- **126 unique heroes** with full team compositions
- **Features**: hero compositions, game metadata, skill brackets
- **Target**: binary classification (Radiant win/loss)
- **Current baseline**: 55.45% accuracy (vs 53.48% majority class)

## Key Findings

### 🎯 **Model Performance**
- **Hero Baseline**: 55.45% accuracy (+3.7% improvement over baseline)
- **Best Model**: Logistic Regression (outperformed Random Forest)
- **AUC Score**: 0.577 - solid predictive power

### 📊 **Hero Meta Insights**
**Strongest Heroes** (High Winrate):
- Meepo: 54.7% winrate
- Vengeful Spirit: 54.7% winrate  
- Wraith King: 54.5% winrate

**Weakest Heroes** (Low Winrate):
- Doom: 42.5% winrate
- Monkey King: 43.5% winrate
- Bristleback: 44.4% winrate

**Most Popular Heroes**:
- Pudge: 2.63% pick rate
- Lion: 2.61% pick rate
- Axe: 2.19% pick rate

### ⚖️ **Game Balance**
- **Radiant Advantage**: 53.48% winrate (3.48% deviation from balanced)
- **Consistent across ranks**: 51-54% Radiant winrate at all skill levels
- **Average match duration**: 41.8 minutes

## Setup

### 1. **Clone and setup environment:**
```bash
git clone <your-repo-url>
cd dota-ml
conda env create -f environment.yml
conda activate dota-ml
```

### 2. **Setup dotaconstants submodule:**
```bash
git submodule update --init --recursive
cd libs/dotaconstants
npm install
npm run build
cd ../..
```

### 3. **Add your data:**
   - Place `public_matches_500k.json` in `./data/` directory
   - Get data from: `curl "https://api.opendota.com/api/explorer" --get --data-urlencode "sql=SELECT * FROM public_matches ORDER BY match_id DESC LIMIT 500000"`

### 4. **Explore the dataset:**
```bash
cd tools
python explore_dataset.py
```

### 5. **Train baseline model:**
```bash
cd tools
python hero_baseline.py
```

## Project Structure
```
dota2-ml/
├── data/                          # Dataset files (not in git)
│   ├── public_matches_500k.json   # Raw OpenDota data
│   └── processed/                 # Processed ML-ready data
│       ├── hero_baseline_train.csv.gz
│       ├── hero_baseline_val.csv.gz
│       ├── hero_baseline_test.csv.gz
│       ├── hero_baseline_logistic_regression.pkl
│       └── hero_baseline_metadata.json
├── libs/                          # External dependencies
│   └── dotaconstants/             # Hero names and game constants (submodule)
├── src/                           # Source code (future models)
├── tools/                         # Analysis and ML scripts
│   ├── explore_dataset.py         # Enhanced data exploration
│   └── hero_baseline.py           # Hero one-hot baseline model
├── notebooks/                     # Jupyter notebooks (future analysis)
├── models/                        # Trained models (not in git)
├── results/                       # Analysis outputs (not in git)
│   ├── time_analysis.png
│   ├── enhanced_hero_analysis.png
│   └── duration_analysis.png
├── environment.yml                # Conda environment
├── requirements.txt               # Pip requirements
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Features

### **Current Implementation (Hero Baseline)**
- **Hero one-hot encoding**: 252 binary features (126 heroes × 2 teams)
- **Target variable**: Radiant win (1) vs Dire win (0)
- **Models**: Logistic Regression, Random Forest
- **Data split**: 60% train, 20% validation, 20% test

### **Feature Engineering Pipeline**
1. **Hero Features**: Binary encoding for each hero on each team
2. **Data Validation**: Team composition verification (5v5)
3. **Train/Val/Test Split**: Stratified splits maintaining class balance
4. **Model Comparison**: Multiple algorithms with performance metrics

## Results & Analysis

### **Model Performance**
```
Logistic Regression:  55.45% accuracy (AUC: 0.577)
Random Forest:        54.77% accuracy (AUC: 0.567)
Baseline (majority):  53.48% accuracy
```

### **Feature Importance Insights**
The model correctly identified heroes with extreme winrates:
- **Highest Impact**: Doom, Monkey King (low winrate heroes)
- **Positive Impact**: Vengeful Spirit, Wraith King (high winrate heroes)
- **Linear Relationships**: Logistic Regression outperformed Random Forest, suggesting additive hero effects

### **Statistical Significance**
- **500k matches**: Large enough for reliable statistical inferences
- **126 heroes**: Comprehensive hero pool coverage
- **Balanced validation**: Consistent class distribution across splits

## Next Development Steps

### **Short Term (Immediate)**
1. ✅ **Hero baseline established** (55.45% accuracy)
2. 🔄 **Team composition features** (STR/AGI/INT counts, melee/ranged ratios)
3. 🔄 **Game metadata features** (rank, game mode, time of day)
4. 🔄 **Advanced models** (XGBoost, LightGBM, Neural Networks)

### **Medium Term (Iterative Improvements)**
1. 🔄 **Hero synergy features** (statistically significant pairs only, min 200+ games)
2. 🔄 **Role-based features** (carry+support combinations, initiator+nuker)
3. 🔄 **Hyperparameter optimization** (Grid search, Bayesian optimization)
4. 🔄 **Ensemble methods** (Model stacking, voting classifiers)

### **Long Term (Production)**
1. 🔄 **Real-time prediction API** (Flask/FastAPI service)
2. 🔄 **Web interface** (Hero selection → winrate prediction)
3. 🔄 **Live data integration** (OpenDota API streaming)
4. 🔄 **Model monitoring** (Performance tracking, data drift detection)

## Usage Examples

### **Basic Exploration**
```bash
# Explore your dataset with enhanced analysis
cd tools
python explore_dataset.py

# Output: Hero winrates, synergies, temporal patterns, visualizations
```

### **Train Baseline Model**
```bash
# Train hero one-hot baseline
cd tools
python hero_baseline.py

# Output: Trained models, feature importance, performance metrics
```

### **Load Trained Model**
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('data/processed/hero_baseline_logistic_regression.pkl')

# Load test data
test_data = pd.read_csv('data/processed/hero_baseline_test.csv.gz')
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

## Technical Notes

### **Data Quality**
- **100% valid matches**: All matches have proper 5v5 team compositions
- **No missing data**: Complete hero and metadata information
- **Recent meta**: Data from August 2025, reflects current game balance

### **Model Validation**
- **Stratified splits**: Maintain class balance across train/val/test
- **Cross-validation ready**: Consistent random seeds for reproducibility
- **Performance tracking**: Comprehensive metrics (accuracy, AUC, feature importance)

### **Computational Requirements**
- **Training time**: ~2-3 minutes on standard laptop
- **Memory usage**: ~200MB for full dataset
- **Storage**: ~80MB compressed CSV files

## Known Limitations

1. **Hero synergies**: Current baseline doesn't capture hero interactions
2. **Game duration**: No temporal features (early/late game preference)
3. **Player skill**: Individual player performance not included
4. **Item builds**: No item/ability progression data
5. **Patch changes**: Model trained on single patch timeframe

## Contributing

1. **Data collection**: Update with newer matches
2. **Feature engineering**: Add team composition, synergy features
3. **Model improvements**: Experiment with advanced algorithms
4. **Production deployment**: Build prediction interface

## References

- **OpenDota API**: https://docs.opendota.com/
- **Dota Constants**: https://github.com/odota/dotaconstants
- **Hero Data**: Comprehensive hero statistics and metadata
- **Game Balance**: Current patch 7.37 meta analysis

---

**Status**: ✅ **Baseline Complete** | 🔄 **Feature Engineering in Progress** | 🎯 **Target: 60%+ Accuracy**