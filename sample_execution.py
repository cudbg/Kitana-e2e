import pandas as pd
import numpy as np
from search_engine.experiment import ScaledExperiment
from search_engine.config import get_config, Config, DataConfig, SearchConfig, ExperimentConfig, LoggingConfig
from search_engine.entity_linking.el_test import DBpediaLinker, TURLExecuter, PredictionEntityConverter
import os
import json
import argparse

def plot_two_experiments_results_max(
    origin_result: dict, 
    output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"] 
    max_len = len(acc1)
    acc1_padded = list(acc1) + [np.nan] * (max_len - len(acc1))

    iterations = range(1, max_len + 1)
    df = pd.DataFrame({
        "iteration": iterations,
        "origin_exp": acc1_padded,
    })


    from search_engine.utils.plot_utils import plot_whiskers
    plot_whiskers(
        df=df,
        x_col="iteration", 
        y_cols=["origin_exp"],
        labels=["Origin Exp"],
        colors=["blue"],
        linestyles=["-"],
        figsize=(8, 6),
        resultname=output_fig,
        xlabel="Iteration",
        ylabel="Accuracy"
    )

    return df

config1 = Config(
    search=SearchConfig(iterations=12),
    data=DataConfig(
        directory_path='data/country_extend_table_search/seller',
        buyer_csv='data/country_extend_table_search/buyer/master.csv',
        join_keys=[['Country'], ['year']],
        target_feature='suicides_no',
        one_target_feature=False,
        need_to_clean_data=True
    ),
    experiment=ExperimentConfig(
        plot_results=True,
        results_dir='results/'
    ),
    logging=LoggingConfig(
        level='ERROR',
        file='logs/experiment.log'
    )
)
# Run exps
company = ScaledExperiment(config1)

company_experiment_result = company.run()

# Return as list
results_list = [company_experiment_result]

# plot
plot_two_experiments_results_max(
    origin_result=company_experiment_result,
    output_fig="results/comparison_country_extend_table_search_whiskers.png"
)