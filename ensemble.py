"""Evaluate a probability ensemble from multiple training result folders."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("folders",nargs="+")
    parser.add_argument("--model",default="jac_scm")
    parser.add_argument("--output",default="results_ensemble")
    args=parser.parse_args(); probabilities=[]; labels=None
    for folder in map(Path,args.folders):
        current=np.load(folder/f"{args.model}_probabilities.npy")
        current_labels=np.load(folder/f"{args.model}_test_labels.npy")
        if labels is not None and not np.array_equal(labels,current_labels):
            raise ValueError("Ensemble members must use the same test split")
        labels=current_labels; probabilities.append(current)
    pred=np.mean(probabilities,axis=0).argmax(1)
    precision,recall,f1,_=precision_recall_fscore_support(labels,pred,average="macro",zero_division=0)
    output=Path(args.output); output.mkdir(exist_ok=True)
    np.savetxt(output/"confusion.csv",confusion_matrix(labels,pred),fmt="%d",delimiter=",")
    print({"accuracy":accuracy_score(labels,pred),"precision":precision,"recall":recall,"f1":f1})


if __name__=="__main__": main()
