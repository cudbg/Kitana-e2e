#!/bin/bash

experiments=("run_company_exp" "run_university_exp" "run_fifa_exp" "run_book_exp" "run_currency_exp")

for exp in "${experiments[@]}"; do
    echo "Running experiment: $exp"
    python main.py --exp "$exp"
done
