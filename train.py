"""Train one or all models reported in the paper."""
from __future__ import annotations
import argparse, copy, csv, json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from model import build_neural_model, make_lawd_adamw
from preprocessing import cars_select, msc

NEURAL = ["lenet5", "lenet5_ssc_jam", "lenet5_lawd", "jac_scm", "lstm", "tcn", "resnet", "transformer"]
CLASSICAL = ["knn", "bp", "tree", "nb", "svm", "gbdt"]
ALL_MODELS = NEURAL + CLASSICAL


class SpectralDataset(Dataset):
    def __init__(self, x, y, augment=False, noise=.015):
        self.x=torch.tensor(x,dtype=torch.float32); self.y=torch.tensor(y); self.augment=augment; self.noise=noise
    def __len__(self): return len(self.y)
    def __getitem__(self,index):
        x=self.x[index].clone()
        if self.augment:
            x = x * (1 + torch.randn(1).item()*.025) + torch.randn(1).item()*.02
            x = x + torch.randn_like(x)*self.noise
        return x,self.y[index]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f}


def load_data(path, seed, label_column="auto", feature_count=256, cars_iterations=30):
    data = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
    header = [str(value).lstrip("\ufeff").strip() for value in
              np.genfromtxt(path, delimiter=",", dtype=str, max_rows=1).tolist()]
    if label_column == "auto":
        label_index = 0 if str(header[0]).lower() in {"origin", "label", "class", "target"} else len(header)-1
    else:
        if label_column not in header: raise ValueError(f"Label column {label_column!r} not found in CSV")
        label_index = header.index(label_column)
    labels = data[:, label_index]; x = np.delete(data, label_index, axis=1).astype(np.float64)
    if not np.isfinite(x).all(): raise ValueError("Spectral matrix contains missing or non-finite values")
    encoder = LabelEncoder(); y = encoder.fit_transform(labels)
    calibration, test = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=seed)
    train, valid = train_test_split(calibration, test_size=.2, stratify=y[calibration], random_state=seed)
    x_cal, ref = msc(x[calibration]); lookup = {idx: row for idx, row in zip(calibration, x_cal)}
    x_test, _ = msc(x[test], ref)
    selected = np.arange(x.shape[1]); feature_count = min(feature_count, x.shape[1])
    if x.shape[1] > feature_count:
        selected, history = cars_select(x_cal, y[calibration], feature_count, iterations=cars_iterations,
                                        seed=seed, return_history=True)
        np.savetxt("cars_indices.csv", selected, fmt="%d", delimiter=",")
        with open("cars_history.csv", "w", newline="") as f: csv.writer(f).writerows([["iteration","variables","rmsecv"], *history])
    return (np.array([lookup[i] for i in train])[:, selected], y[train],
            np.array([lookup[i] for i in valid])[:, selected], y[valid], x_test[:, selected], y[test], encoder.classes_)


def train_neural(name, data, args, out):
    xt, yt, xv, yv, xs, ys, classes = data; device = torch.device(args.device)
    scaler = StandardScaler().fit(xt); xt, xv, xs = scaler.transform(xt), scaler.transform(xv), scaler.transform(xs)
    model = build_neural_model(name, xt.shape[1], len(classes)).to(device)
    use_lawd = name in {"lenet5_lawd", "jac_scm"}
    optimizer = make_lawd_adamw(model, args.lr) if use_lawd else torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.restart_period, T_mult=2)
    counts = np.bincount(yt, minlength=len(classes)); weights = len(yt)/(len(classes)*counts)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(weights), dtype=torch.float32, device=device),
                                  label_smoothing=args.label_smoothing)
    dataset = SpectralDataset(xt,yt,augment=args.augment and name=="jac_scm",noise=args.noise)
    sampler = WeightedRandomSampler(weights[yt],len(yt),replacement=True) if args.balanced_sampler else None
    loader = DataLoader(dataset,args.batch_size,shuffle=sampler is None,sampler=sampler)
    vx, vy = torch.tensor(xv, dtype=torch.float32, device=device), torch.tensor(yv, device=device)
    best_f1, best_loss, stale, state, history = -1., float("inf"), 0, None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        if epoch < args.warmup_epochs:
            factor = (epoch+1)/args.warmup_epochs
            for group, base in zip(optimizer.param_groups, base_lrs): group["lr"] = base*factor
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(loss.item())
        if epoch >= args.warmup_epochs: scheduler.step(epoch-args.warmup_epochs)
        model.eval()
        with torch.no_grad():
            val_logits=model(vx); val_loss=loss_fn(val_logits,vy).item(); val_pred=val_logits.argmax(1).cpu().numpy()
        val_f1=metrics(yv,val_pred)["f1"]
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":val_loss,"val_f1":val_f1})
        if val_f1 > best_f1 + 1e-6 or (abs(val_f1-best_f1)<1e-6 and val_loss < best_loss):
            best_f1, best_loss, stale, state = val_f1, val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(xs, dtype=torch.float32, device=device)),1).cpu().numpy()
        pred = probabilities.argmax(1)
    torch.save({"model":state,"classes":classes.tolist(),"selected_features":xt.shape[1]}, out/f"{name}.pt")
    (out/f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savetxt(out/f"{name}_confusion.csv", confusion_matrix(ys,pred), fmt="%d", delimiter=",")
    np.save(out/f"{name}_probabilities.npy",probabilities)
    np.save(out/f"{name}_test_labels.npy",ys)
    return metrics(ys,pred)


def train_classical(name, data, args, out):
    xt, yt, xv, yv, xs, ys, _ = data; xfit=np.vstack([xt,xv]); yfit=np.r_[yt,yv]
    scaler=StandardScaler().fit(xfit); xfit, xs=scaler.transform(xfit),scaler.transform(xs)
    models={"knn":KNeighborsClassifier(),"bp":MLPClassifier(hidden_layer_sizes=(128,64),max_iter=1000,random_state=args.seed),
            "tree":DecisionTreeClassifier(random_state=args.seed),"nb":GaussianNB(),"svm":SVC(),
            "gbdt":GradientBoostingClassifier(random_state=args.seed)}
    pred=models[name].fit(xfit,yfit).predict(xs); np.savetxt(out/f"{name}_confusion.csv",confusion_matrix(ys,pred),fmt="%d",delimiter=",")
    return metrics(ys,pred)


def main():
    default_csv = Path("dataset") / "20251203_535_产地_7.csv"
    p=argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default=str(default_csv),
                   help=f"spectral CSV path (default: {default_csv})")
    p.add_argument("--model",choices=ALL_MODELS+["all"],default="jac_scm")
    p.add_argument("--epochs",type=int,default=300); p.add_argument("--patience",type=int,default=20); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--warmup-epochs",type=int,default=5); p.add_argument("--restart-period",type=int,default=10)
    p.add_argument("--label-column",default="auto"); p.add_argument("--features",type=int,default=272); p.add_argument("--cars-iterations",type=int,default=30)
    augment_group=p.add_mutually_exclusive_group()
    augment_group.add_argument("--augment",dest="augment",action="store_true")
    augment_group.add_argument("--no-augment",dest="augment",action="store_false")
    p.set_defaults(augment=True); p.add_argument("--noise",type=float,default=.015)
    sampler_group=p.add_mutually_exclusive_group()
    sampler_group.add_argument("--balanced-sampler",dest="balanced_sampler",action="store_true")
    sampler_group.add_argument("--no-balanced-sampler",dest="balanced_sampler",action="store_false")
    p.set_defaults(balanced_sampler=False)
    p.add_argument("--label-smoothing",type=float,default=.05)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--split-seed",type=int,default=42)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output",default="results")
    args=p.parse_args(); seed_all(args.seed); out=Path(args.output); out.mkdir(exist_ok=True)
    data=load_data(args.csv,args.split_seed,args.label_column,args.features,args.cars_iterations)
    print(f"data: train={len(data[1])}, valid={len(data[3])}, test={len(data[5])}, features={data[0].shape[1]}, classes={data[6].tolist()}")
    names=ALL_MODELS if args.model=="all" else [args.model]; rows=[]
    for name in names:
        seed_all(args.seed); result=train_neural(name,data,args,out) if name in NEURAL else train_classical(name,data,args,out)
        rows.append({"model":name,**result}); print(name, {k:round(v,4) for k,v in result.items()})
    with open(out/"results_summary.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__=="__main__": main()
