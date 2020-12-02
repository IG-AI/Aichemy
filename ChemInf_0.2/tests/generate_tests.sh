#!/bin/bash

export PYTHONHASHSEED=0

pynguin --algorithm WSPY --project_path ./ --output_path ./tests/units --report_dir ./logs/pynguin/ \
--module_name tests.integrations.all_modules --seed 123 -v

wait