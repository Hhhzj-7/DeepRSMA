# DeepRSMA
DeepRSMA: a cross-fusion based deep learning method for RNA-Small molecule binding affinity prediction

## Data
All the data used in this study are provided in the data floder.
* RSM_data: Raw data for CV.
* blind_test: Raw data for blind test.
* independent_data.csv: Raw data for independent test.
* representation_cv: pre-trained RNA embedding for CV.
* representation_independent: pre-trained RNA embedding for independent test.
* RNA_contact: RNA contact map.

## Environment
`You can create a conda environment for DeepRSMA by ‘conda env create -f environment.yml‘.`

## Train on multiple experimental settings
* Cross validation: `‘python main_cv.py‘`
* Blind test: `‘python main_blind.py‘`
* Independent test: `‘python main_independent.py‘` 