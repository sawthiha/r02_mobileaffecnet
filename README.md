# MobileAffectNet Research Materials

This repository contains research related materials for MobileAffectNet model.

## Environment Setup
You can use conda to setup the Python environment by executing
```sh
conda env create -f conda.yml
```

## Evaluation Dataset
You can request the AffectNet dataset from the original authors using the link [http://mohammadmahoor.com/affectnet-request-form/](http://mohammadmahoor.com/affectnet-request-form/).

The default (expected) path for the validation dataset is `./val_set`. However, you may edit the `VALIDATION_PATH` from `evaluate.py` if the validation dataset is stored somewhere else.

## Evaluation
Given the Python environment and the AffecNet dataset, you can simply run the `evaluate.py`
```
python3 evaluate.py
```