import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torch.utils.data import DataLoader

from models.GIN import GIN
from models.MLP import MLP
from train_model import train, evaluate
from utils.attack_utils import *
from utils.data_utils import collate_graphs, collate_tensors, get_data_from_dataset


def evaluate_model(model_name, dataset, fold=10, times=10, epoches=300):
    # config GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load data
    if model_name in ['mlp', 'rf', 'svm', 'gbdt']:
        data, labels, num_classes, X = get_data_from_dataset(dataset, convert2tensor=True)
        collate_fn = collate_tensors
    else:
        data, labels, num_classes = get_data_from_dataset(dataset, convert2tensor=False, node_importance=True)
        collate_fn = collate_graphs
    scores = []
    for time in range(times):
        valid_curves = []
        # K-fold split
        kf = StratifiedKFold(n_splits=fold, shuffle=True)
        cv = 0
        if model_name == 'rf':
            rfc = RandomForestClassifier(oob_score=True)
            cv_score = cross_val_score(rfc, X, labels.squeeze(), cv=kf, scoring='accuracy')
            scores.append(cv_score)
            print(np.mean(scores))
            continue
        if model_name == 'gbdt':
            gbdt = RandomForestClassifier(oob_score=True)
            cv_score = cross_val_score(gbdt, X, labels.squeeze(), cv=kf, scoring='accuracy')
            scores.append(cv_score)
            print(np.mean(scores))
            continue

        for train_index, test_index in kf.split(data, labels):
            cv += 1
            data_train, data_test = data[train_index], data[test_index]
            # len_train = int(len(data_train) * 0.9)
            train_loader = DataLoader(data_train, batch_size=128, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(data_test, batch_size=128, shuffle=False, collate_fn=collate_fn)
            # valid_loader = DataLoader(data_train[len_train:], batch_size=256, shuffle=False, collate_fn=collate_fn)
            if model_name == 'mlp':
                model = MLP(
                    input_size=93,
                    output_size=num_classes
                )
            if model_name == 'gin':
                model = GIN(
                    input_dim=3,
                    hidden_dim=16,
                    output_dim=num_classes
                )
            save_path = f'./checkpoints/{model_name}_{dataset}'
            _, valid_curve = train(
                device=device,
                model=model,
                save_path=save_path,
                train_loader=train_loader,
                val_loader=test_loader,
                max_epoch=epoches
            )
            valid_curves.append(valid_curve)
            model = torch.load(save_path)
            acc = evaluate(test_loader, device, model)
            scores.append(acc)
            print(scores)
        np.save(f'./accuracy/valid_{model_name}_{dataset}', np.array(valid_curves))
    np.save(f'./accuracy/score_{model_name}_{dataset}', np.array(scores))
