# DeepRSMA
DeepRSMA: a cross-fusion based deep learning method for RNA-Small molecule binding affinity prediction

by: Zhijian Hunag, Yucheng Wang, Song Chen, Yaw Sing Tan, Lei Deng and Min Wu

## Data
All the data used in this study data processing code are provided in the data floder.
* RSM_data: Raw data for cross validation.
* blind_test: Raw data for blind test.
* independent_data.csv: Raw data for independent test.
* representation_cv: pre-trained RNA embedding for cross validation.
* representation_independent: pre-trained RNA embedding for independent test.
* RNA_contact: RNA contact map.
* process_data_molecule.py: Molecular data processing code for cross validation and blind test.
* process_data_rna.py: RNA data processing code  for cross validation and blind test.
* process_independent_mole.py: Molecular data processing code for independent test.
* process_independent_rna.py: RNA data processing code for independent test.

The Generation of contact map and pre-trained RNA embedding relies on the following libraries:
* SPOT-RNA-2D (https://github.com/jaswindersingh2/SPOT-RNA-2D)
* RNA-FM (https://github.com/soedinglab/CCMpred)

## Environment
You can create a conda environment for DeepRSMA by `‘conda env create -f environment.yml‘.`

## Train on multiple experimental settings
In this study, we employed the following 3 different experimental setups.
* Cross validation: `‘python main_cv.py‘`
* Blind test: `‘python main_blind.py‘`
* Independent test: `‘python main_independent.py‘`

## Contact
If you have any issues or questions about this paper or need assistance with reproducing the results, please contact me.

Zhijian Huang

School of Computer Science and Engineering,

Central South University

Email: zhijianhuang@csu.edu.cn
