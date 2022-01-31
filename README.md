## Requirements
- pytorch
- scikit-learn
- networkx
- optuna
- tqdm

## Run codes
### Node Classification
python run_node_cls.py --rate 0.8 --dataset cora --model recursive

### Link Prediction
python run_link_pred.py --dataset citeseer --type bias --rate 0.5
