import numpy as np
import torch
from model import build_neural_model, make_lawd_adamw
from preprocessing import cars_select, msc
from train import load_data


def test_all_neural_shapes():
    for name in ["lenet5","lenet5_ssc_jam","lenet5_lawd","jac_scm","lstm","tcn","resnet","transformer"]:
        assert build_neural_model(name)(torch.randn(3,114)).shape == (3,10)
    assert build_neural_model("jac_scm",272,7)(torch.randn(3,272)).shape == (3,7)


def test_lawd_groups_and_preprocessing():
    model=build_neural_model("jac_scm"); opt=make_lawd_adamw(model)
    assert len(opt.param_groups)==3
    rng=np.random.default_rng(4); x=rng.normal(size=(40,120)); y=np.repeat(np.arange(4),10)
    corrected,ref=msc(x); assert corrected.shape==x.shape and ref.shape==(120,)
    selected=cars_select(corrected,y,n_features=20,iterations=5,components=3)
    assert selected.shape==(20,) and len(np.unique(selected))==20


def test_first_column_label_loading(tmp_path):
    path=tmp_path/"spectra.csv"
    rows=["\ufefforigin,"+",".join(f"f{i}" for i in range(20))]
    for label in range(3):
        for sample in range(10): rows.append(str(label)+","+",".join(str(label+sample*.01+i*.001) for i in range(20)))
    path.write_text("\n".join(rows),encoding="utf-8")
    data=load_data(path,42,feature_count=10,cars_iterations=3)
    assert data[0].shape[1]==10 and data[6].tolist()==["0","1","2"]
