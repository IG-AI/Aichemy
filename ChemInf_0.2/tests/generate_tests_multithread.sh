#!/bin/bash

export PYTHONHASHSEED=0

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.classifiers_module --seed 123 -v &

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.model_module --seed 123 -v &

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.postprocessing_module --seed 123 -v &

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.preprocessing_module --seed 123 -v &

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.operator_module --seed 123 -v &

wait