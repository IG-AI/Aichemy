## Preprocessing

All of the amcp_main.py script modes reads a white space delimited file where the first column is the sample ID, the second is the sample's class, and the following fields are the descriptive features of the sample. The features can be either floating point numbers or integers. 

```
CP001892232805_-27.90 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 ... 0 0
CP002491258647_-28.51 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
CP001136835595_-34.41 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
CP002361737404_-34.11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
CP000831597329_-32.93 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ... 0 1
CP000209597191_-33.22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ... 0 0
CP000362090899_-32.33 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
CP002569931590_-26.92 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ... 0 0
CP003236811467_-40.40 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 ... 0 0
CP002361189791_-33.72 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 ... 0 0
```

### 


### Morgan Fingerprints

To be able to use the energy plotting scripts. It is important to append the scores to the identifiers.

In the above example, morgan fingerprints (ECFPs) are used as features. Morgan fingerprints can be obtained from an .ism file. An .ism file will contain SMILES representations of molecules as the first field with ID and properties as the second field. 


example.ism:
```
Cc1ccc(cc1Cl)C=CC(=O)N2CCC(C2)(C)CNc3c(nccn3)C#N CP000469650145 -63.64
Cc1c(cc(cn1)NC(=O)C)C(=O)N2CCN(CC23CCC3)C(=O)C=CC(=O)N CP000105192859 -59.13
CC(C1CN(CCO1)C(=O)c2cc3cc[nH]c(=O)c3nc2)NC(=O)c4[nH]ccn4 CP000261423010 -58.56
CC1CN(CCN1C(=O)c2cc(cnc2)N)C(=O)c3ccc(cc3)OC4CCC4 CP000121223014 -58.39
Cc1cccc(c1)CC(CNCc2ccccc2n3cncn3)C(=O)O CP001507138880 -58.14
c1ccc(cc1)CN(CC2(CN(C2)CC#N)O)C(=O)Cc3c[nH]c4c3cccn4 CP000130574827 -57.96
C[C@H]1CN(CCN1C(=O)c2cncc3c2cc[nH]3)C(=O)c4cnc5c(n4)cc[nH]5 CP000377639127 -57.88
Cc1ccc(c(c1)Cl)CC(=O)N(C)CCN(C)C(=O)Cc2cc(cnc2)O CP002093680592 -57.87
Cc1ccc(cc1OC)C(=O)NCC2CCCN(C2)C(=O)Cc3cc(cnc3)O CP000342807662 -57.82
CC(C(=O)NCC1(CCN(CC1)C(=O)c2[nH]ccn2)c3ccccc3)NC(=O)N CP000050522257 -57.82
```

Here, the 2nd column is the ID of the sample and the 3rd column is the docking score. Append the docking score to the ID to keep track of it for subsequent plotting.

```bash
awk '{print $1,$2"_"$3}' example.ism > reformatted_example.ism
```

To generate the fingerprints, you will need the FRONTLINE singularity image. Use the following command:
```
module use /proj/carlssonlab/envmod
module load FRONTLINE
```

```bash
$FRONTENV python3 amcp_0.1/scripts/morgan_features.py -i reformatted_example.ism -o example.1024r2.csv -r2
```

This will generate features with radius 2 (ECFP4) and 1024 bits. Depending on the size of your file, you can parallelize this with the following approach.

```bash
split example.ism example.split_ -d -a 3 -l 2000000
ls example.split_* > filelist
```

This splits the ism file into separate files of 2 million samples each. You can then copy the submit_morgan_array.sh from amcp_0.1/scripts to your working directory and submit it.

### Docking Score

In the current state of the pipeline, the docking threshold for the best scoring 1 % molecules can be found easiest by sorting the scores and using sed.

From the example.ism file, use the following commands:

```bash
awk '{print $3}' > scores 
sort scores > sorted_scores
wc -l sorted_scores # Will write out the number of lines
sed 'NUMq;d' # Replace NUM with 0.01 * the line numbers from wc -l
```

This will give you the docking score threshold. For example -33.84.

To assign classes to the samples. Use the assign-multiclass-ls-compress.py script. Here you do not need the frontline environment.

```
python3 amcp_0.1/scripts/assign-multiclass-ls-compress.py -i example.csv -o example.class_-33.84.csv -c -33.84
```

If you want to assign several classes based on thresholds, this can be done by simply adding more cutoffs to the -c flag.

If the samples are not labeled, you can assign all samples '0' as a dummy class.

```bash
sed 's/^[^ ]*/& 0/' example.csv > example.dummyclass.csv
```

## Build

Training new models is performed with the build mode. Example:

```
$FRONTENV python3 amcp_0.1/amcp -m build -i example.class_-33.84.csv -md example_amcp_models_directory -p parameters.txt
```

Briefly, the way the build mode works is by reading in the entire input file into memory. It will then split the file randomly into a proper training set and a calibration set. The ratio is specified by the prop_train_ratio parameter. The underlying classifier is trained on the proper training set. The calibration set is predicted with the model and the samples nonconformity scores are calculated for all classes. This process is repeated in sequence for the number of models specified by the nr_of_build_models parameter. Lastly, the models and the calibration samples' nonconformity scores are saved to the directory specified by the -md flag.

### Parallelization 

Parallelization of the build mode can be performed by building several models separately, with the same input file, and then bringing them together into one directory. The naming of the models is important. To adhere to the convention expected by subsequent prediction, the separate directories should be placed under a master directory. All the models will then be put in this directory with the correct names.

Create a folder structure such as this:
```
amcp_models/
├── amcp_models_0
├── amcp_models_1
├── amcp_models_2
├── amcp_models_3
└── amcp_models_4
```
Write the path to each directory to a 'dirlist' file. Using this approach, you can reduce the number of build models, and instead perform it in parallel using the submit_amcpbuild_array.sh script.  In the above example, dirlist would be a textfile with the following content.

```
amcp_models/amcp_models_0
amcp_models/amcp_models_1
amcp_models/amcp_models_2
amcp_models/amcp_models_3
amcp_models/amcp_models_4
```


 After all models are built, you can aggregate them:

```bash
cd  amcp_models
bash /amcp_0.1/scripts/amcp_aggregate_models.sh
```

Note: This would also work if different training files had been used to construct the models. In this way, more samples can be used as training data without having to read them all into memory. In the literature, this is referred to as synergy conformal prediction.

## Predict

To use built models for prediction, you run amcp with the predict mode. Example:

```
$FRONTENV python3 amcp_0.1/amcp -m predict -i example.dummyclass.csv -md example_amcp_models_directory -p parameters.txt -o example.dummyclass.prediction.csv
```

This will write out a prediction file with two headers followed by the predicted samples. Example:

```
amcp_validation	test_samples	validation_file:"test_features2.csv"
sampleID	real_class	p(0)	p(1)
CP001296598236_-23.39	0	0.41556937772965746	0.12869102194587417
CP000210536053_-24.77	0	0.7719261806095615	0.0095168103150632
CP001386728846_-41.38	0	0.17636899040586995	0.35040098850941304
CP003074914220_-26.24	0	0.23689406398558713	0.3333333333333333
CP002037258855_-29.03	0	0.29786095338082125	0.16
CP003012672169_-17.04	0	0.2166957792905326	0.3306332456600968
CP001013145147_-20.50	0	0.2414891107989777	0.2727272727272727
CP000015740219_-29.51	0	0.47331032545952423	0.13189965193615916
```

The prediction mode works by first loading in each model and calibration sets' nonconformity scoresinto memory. The input file is read in blocks. For each block, the samples are predicted with each model. The conformal prediction p-values are calculated by determining where a given sample's nonconformity score would be placed in the calibration set's nonconformity score vector. This is done for each class (mondrian approach). The median of the p-values are then outputted as the p-value.

### Parallelization

Parallelization is performed by splitting the input files in a similar fashion as in the preprocessing section. The paths to the files should then be given in the predfilelist, similarly as the dirlist for building. You can then use the submit_amcppred_array.sh script.

The prediction files can be concatenated with tail. This will skip the header lines.

```
tail -n+3 example.pred_split* > predictions.csv
```

## Validation

The validation mode is run using the common K-fold cross validation. Example:

```
$FRONTENV python3 amcp_0.1/amcp -m validation -i example.csv -o example.val_pred.csv
```

The validation mode reads in the input file and splits it in K (specified by the val_fold parameter) equal parts. Aggregated models (nr specified by the nr_of_build_models parameter) are built on K-1 parts and The holdout set is predicted. This is repeated until all samples in the file are predicted. It is important to keep in mind that the actual training size is (K-1) * training set size. None of the validation models are saved on disk. 

### Significance Testing

Significance testing for different classifiers or parameters can be performed on the prediction files with the amcp_mcnemar_test.py script. You specify the two files and the corresponding epsilon thresholds that you want to consider. Example:

```
$FRONTENV python3 amcp_0.1/scripts/amcp_mcnemar_test.py -i1 example1.pred.csv -s1 0.2 -i2 example2.pred.csv -s2 0.2 > stat_report.txt
```

This will test for significant differences in recall (sensitivity), specificity and accuracy for the single label predictions. The test is paired. This means that the prediction files must contain the same samples. The script also expects that they are in the same order. It can also be used on the prediction files obtained from the predict mode.


## Postprocessing

To evaluate the predictions, the amcp_summarize_pred.py script is used:

```
$FRONTENV python3 amcp_0.1/scripts/amcp_summarize_pred.py -i example.pred.csv -o example.pred.summary.csv
```

The statistics of the summary file can also be plotted with the amcp_plot_summary.py script which is going to spit out a bunch of png files, so it is best to do it locally.

Extraction of {1} set can be performed with awk:

```
awk '$3<0.2 && $4>0.2' example.pred.csv > example.pred.1set.csv
``` 

