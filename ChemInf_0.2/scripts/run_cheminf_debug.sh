#!/bin/bash

model_dir="cheminf/data/cheminf_models/debug_nn"
prediction_dir="cheminf/data/cheminf_predictions/debug"

echo "-------------------------------------------"
echo "DEBUGGING BUILD/NN"
echo "-------------------------------------------"
python cheminf/ build -cl nn -i cheminf/data/d2.trailset.balanced.train.csv
python cheminf/ build -cl nn -n debug_nn -i cheminf/data/d2.trailset.balanced.train.csv
python cheminf/ build -cl nn -i cheminf/data/d2.trailset.balanced.train.csv -md $model_dir

echo "-------------------------------------------"
echo "DEBUGGING PREDICT/NN"
echo "-------------------------------------------"
python cheminf/ predict -cl nn -i cheminf/data/d2.trailset.balanced.test.csv
python cheminf/ predict -cl nn -n debug_nn -i cheminf/data/d2.trailset.balanced.test.csv
python cheminf/ predict -cl nn -i cheminf/data/d2.trailset.balanced.test.csv -o $prediction_dir"_nn/prediction_debug_onname.csv"
python cheminf/ predict -cl nn -n debug_nn -i cheminf/data/d2.trailset.balanced.test.csv -o $prediction_dir"_nn/prediction_debug.csv"

echo "-------------------------------------------"
echo "DEBUGGING BUILD/RNDFOR"
echo "-------------------------------------------"
python cheminf/ build -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.train.csv
python cheminf/ build -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.train.csv -md $model_dir

echo "-------------------------------------------"
echo "DEBUGGING PREDICT/RNDFOR"
echo "-------------------------------------------"
python cheminf/ predict -cl rndfor -i cheminf/data/d2.trailset.balanced.test.csv
python cheminf/ predict -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.test.csv
python cheminf/ predict -cl rndfor -i cheminf/data/d2.trailset.balanced.test.csv -o $prediction_dir"_rndfor/prediction_debug_onname.csv"
python cheminf/ predict -cl rndfor -n debug_rndfor-i cheminf/data/d2.trailset.balanced.test.csv -o $prediction_dir"_rndfor/prediction_debug.csv"

echo "-------------------------------------------"
echo "DEBUGGING VALIDATE/RNDFOR"
echo "-------------------------------------------"
python cheminf/ validate -cl rndfor -i cheminf/data/d2.trailset.balanced.csv
python cheminf/ validate -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.csv
python cheminf/ validate -cl rndfor -i cheminf/data/d2.trailset.balanced.csv -o $prediction_dir"_rndfor/validation_predition_debug.csv"
python cheminf/ validate -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.csv -o $prediction_dir"_rndfor/validation_predition_debug.csv"
python cheminf/ validate -cl rndfor -n debug_rndfor -i cheminf/data/d2.trailset.balanced.csv -o2 cheminf/data/validation_train_set.csv

echo "-------------------------------------------"
echo "DEBUGGING UTILS/RESAMPLE"
echo "-------------------------------------------"
python cheminf/ ultils resample -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.balanced.csv
python cheminf/ ultils resample -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.balanced.csv -ch 10000
python cheminf/ ultils resample -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.balanced.csv -ch 10000 -nc 2
python cheminf/ ultils split -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.split1.csv -o2 cheminf/data/d2.trailset.debug.split2.csv
python cheminf/ ultils split -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.split1.csv -o2 cheminf/data/d2.trailset.debug.split2.csv -s -p 0.9
python cheminf/ ultils trim -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.split1.csv
python cheminf/ ultils trim -i cheminf/data/d2.trailset.csv -o cheminf/data/d2.trailset.debug.split1.csv -s -p 0.9
