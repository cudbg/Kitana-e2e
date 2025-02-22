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
    dbpedia_result: dict, 
    turl_10_result: dict,
    turl_100_top10_single_result: dict,
    dbpedia_10_result: dict,
    dbpedia_100_top10_single_result: dict,
    output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"]  # list
    acc2 = dbpedia_result["accuracy"]    # list
    acc3 = turl_10_result["accuracy"]
    acc4 = turl_100_top10_single_result["accuracy"]
    acc5 = dbpedia_10_result["accuracy"]
    acc6 = dbpedia_100_top10_single_result["accuracy"]

    max_len = max(len(acc1), len(acc2), len(acc3), len(acc4), len(acc5), len(acc6))
    acc1_padded = list(acc1) + [np.nan] * (max_len - len(acc1))
    acc2_padded = list(acc2) + [np.nan] * (max_len - len(acc2))
    acc3_padded = list(acc3) + [np.nan] * (max_len - len(acc3))
    acc4_padded = list(acc4) + [np.nan] * (max_len - len(acc4))
    acc5_padded = list(acc5) + [np.nan] * (max_len - len(acc5))
    acc6_padded = list(acc6) + [np.nan] * (max_len - len(acc6))

    iterations = range(1, max_len + 1)
    df = pd.DataFrame({
        "iteration": iterations,
        "origin_exp": acc1_padded,
        "turl_top10_exp": acc2_padded,
        "turl_10_exp": acc3_padded,
        "turl_100_top10_single_exp": acc4_padded,
        "dbpedia_10_exp": acc5_padded,
        "dbpedia_100_top10_single_exp": acc6_padded
    })


    from search_engine.utils.plot_utils import plot_whiskers
    plot_whiskers(
        df=df,
        x_col="iteration", 
        y_cols=["origin_exp", "turl_top10_exp", "turl_10_exp", "turl_100_top10_single_exp", "dbpedia_10_exp", "dbpedia_100_top10_single_exp"],
        labels=["Origin Exp", "turl top 10 exp", "TURL 10 Exp", "TURL 100 top 10 single Exp", "DBpedia 10 Exp", "DBpedia 100 top 10 single Exp"],
        colors=["blue", "red", "green", "yellow", "purple", "orange"],
        linestyles=["-", "-", "-", "-", "-", "-"],
        figsize=(8, 6),
        resultname=output_fig,
        xlabel="Iteration",
        ylabel="Accuracy"
    )

    return df

def plot_two_experiments_results_max_kitana_alite(
    origin_result: dict, 
    alite_result: dict,
    output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"] 
    acc2 = alite_result["accuracy"]
    max_len = max(len(acc1), len(acc2))
    acc1_padded = list(acc1) + [np.nan] * (max_len - len(acc1))
    acc2_padded = list(acc2) + [np.nan] * (max_len - len(acc2))

    iterations = range(1, max_len + 1)
    df = pd.DataFrame({
        "iteration": iterations,
        "origin_exp": acc1_padded,
        "alite_exp": acc2_padded
    })


    from search_engine.utils.plot_utils import plot_whiskers
    plot_whiskers(
        df=df,
        x_col="iteration", 
        y_cols=["origin_exp", "alite_exp"],
        labels=["Origin Exp", "Alite Exp"],
        colors=["blue", "red"],
        linestyles=["-", "--"],
        figsize=(8, 6),
        resultname=output_fig,
        xlabel="Iteration",
        ylabel="Accuracy"
    )

    return df

def run_multiple_experiment():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/country_extend/seller',
            buyer_csv='data/country_extend/buyer/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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

    config2 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='el_data/country_extend/seller/10_top10',
            buyer_csv='el_data/country_extend/buyer/10_top10/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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
    config3 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='el_data/country_extend/seller/10',
            buyer_csv='el_data/country_extend/buyer/10/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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
    # 100_top10_single
    config4 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='el_data/country_extend/seller/100_top10_single',
            buyer_csv='el_data/country_extend/buyer/100_top10_single/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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

    config5 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='el_data/country_extend/seller/dbpedia/10',
            buyer_csv='el_data/country_extend/buyer/dbpedia/10/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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

    config6 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='el_data/country_extend/seller/100_top10_single_dbpedia',
            buyer_csv='el_data/country_extend/buyer/100_top10_single_dbpedia/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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
    origin_experiment = ScaledExperiment(config1)
    dbpedia_experiment = ScaledExperiment(config2)
    turl_10_experiment = ScaledExperiment(config3)
    turl_100_top10_single_experiment = ScaledExperiment(config4)
    dbpedia_10_experiment = ScaledExperiment(config5)
    dbpedia_100_top10_single_experiment = ScaledExperiment(config6)
    
    origin_experiment_result = origin_experiment.run()
    dbpedia_experiment_result = dbpedia_experiment.run()
    turl_10_experiment_result = turl_10_experiment.run()
    turl_100_top10_single_experiment_result = turl_100_top10_single_experiment.run()
    dbpedia_10_experiment_result = dbpedia_10_experiment.run()
    dbpedia_100_top10_single_experiment_result = dbpedia_100_top10_single_experiment.run()

    # Return as list
    results_list = [
        origin_experiment_result, 
        dbpedia_experiment_result, 
        turl_10_experiment_result, 
        turl_100_top10_single_experiment_result,
        dbpedia_10_experiment_result,
        dbpedia_100_top10_single_experiment_result
        ]

    # plot
    plot_two_experiments_results_max(
        origin_result=origin_experiment_result,
        dbpedia_result=dbpedia_experiment_result,
        turl_10_result=turl_10_experiment_result,
        turl_100_top10_single_result=turl_100_top10_single_experiment_result,
        dbpedia_10_result=dbpedia_10_experiment_result,
        dbpedia_100_top10_single_result=dbpedia_100_top10_single_experiment_result,
        output_fig="comparison_whiskers.png"
    )

    return results_list


def plot_two_experiments_results_max(
    origin_result: dict, 
    dbpedia_result: dict, 
    turl_result: dict,
    output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"]  # list
    acc2 = dbpedia_result["accuracy"]    # list
    acc3 = turl_result["accuracy"]

    max_len = max(len(acc1), len(acc2), len(acc3))
    acc1_padded = list(acc1) + [np.nan] * (max_len - len(acc1))
    acc2_padded = list(acc2) + [np.nan] * (max_len - len(acc2))
    acc3_padded = list(acc3) + [np.nan] * (max_len - len(acc3))

    iterations = range(1, max_len + 1)
    df = pd.DataFrame({
        "iteration": iterations,
        "origin_exp": acc1_padded,
        "dbpedia_exp": acc2_padded,
        "turl_exp": acc3_padded
    })


    from search_engine.utils.plot_utils import plot_whiskers
    plot_whiskers(
        df=df,
        x_col="iteration", 
        y_cols=["origin_exp", "dbpedia_exp", "turl_exp"],
        labels=["Origin Exp", "dbpedia exp", "TURL Exp"],
        colors=["blue", "red", "green"],
        linestyles=["-", "-", "-"],
        figsize=(8, 6),
        resultname=output_fig,
        xlabel="Iteration",
        ylabel="Accuracy"
    )

    return df

def run_multiple_experiment_table_search():
    config1 = Config(
        search=SearchConfig(iterations=50),
        data=DataConfig(
            directory_path='data/country_extend_table_search/seller',
            buyer_csv='data/country_extend_table_search/buyer/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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

    config2 = Config(
        search=SearchConfig(iterations=50),
        data=DataConfig(
            directory_path='el_data/country_extend_table_search/seller/100_top10_single_dbpedia',
            buyer_csv='el_data/country_extend_table_search/buyer/100_top10_single_dbpedia/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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
    config3 = Config(
        search=SearchConfig(iterations=50),
        data=DataConfig(
            directory_path='el_data/country_extend_table_search/seller/100_top10_single',
            buyer_csv='el_data/country_extend_table_search/buyer/100_top10_single/master.csv',
            join_keys=[['Country']],
            target_feature='suicides/100k pop',
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
    origin_experiment = ScaledExperiment(config1)
    dbpedia_experiment = ScaledExperiment(config2)
    turl_experiment = ScaledExperiment(config3)
    
    origin_experiment_result = origin_experiment.run()
    dbpedia_experiment_result = dbpedia_experiment.run()
    turl_experiment_result = turl_experiment.run()

    # Return as list
    results_list = [
        origin_experiment_result, 
        dbpedia_experiment_result, 
        turl_experiment_result, 
        ]

    # plot
    plot_two_experiments_results_max(
        origin_result=origin_experiment_result,
        dbpedia_result=dbpedia_experiment_result,
        turl_result=turl_experiment_result,
        output_fig="comparison_whisker_table_search_long_adjusted.png"
    )

    return results_list

def plot_experiments(
    origin_result: dict, 
    output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"]  # list
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

def run_company_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/company_datasets/seller',
            buyer_csv='data/company_datasets/buyer/company_ipo.csv',
            join_keys=[['company']],
            target_feature='IPO Price',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="comparison_company_whiskers.png"
    )

    return results_list

def run_university_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/university_datasets/seller',
            buyer_csv='data/university_datasets/buyer/2024_rankings.csv',
            join_keys=[['name']],
            target_feature='scores_overall',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_university_whiskers.png"
    )

    return results_list

def run_fifa_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/FIFA_datasets/seller',
            buyer_csv='data/FIFA_datasets/buyer/FIFA - 2022.csv',
            join_keys=[['Team']],
            target_feature='Win',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_fifa_whiskers.png"
    )

    return results_list

def run_book_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/books_ISBN_dataset_datasets/seller',
            buyer_csv='data/books_ISBN_dataset_datasets/buyer/Amazon Books Data.csv',
            join_keys=[['title']],
            target_feature='review_count',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_books_whiskers.png"
    )

    return results_list

def run_currency_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/currency_exchange_rates_datasets/seller',
            buyer_csv='data/currency_exchange_rates_datasets/buyer/Exchange_Rates.csv',
            join_keys=[['Currency']],
            target_feature='Exchange',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_currency_whiskers.png"
    )

    return results_list
    
def run_movie_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/IMDb_movie_dataset_datasets/seller',
            buyer_csv='data/IMDb_movie_dataset_datasets/buyer/data.csv',
            join_keys=[['Movie']],
            target_feature='Movie Rating',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_movie_whiskers.png"
    )

    return results_list

def run_zip_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/US_ZIP_codes_dataset_datasets/seller',
            buyer_csv='data/US_ZIP_codes_dataset_datasets/buyer/USWagesByZip.csv',
            join_keys=[['Zip Code']],
            target_feature='TotalWages',
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
    plot_experiments(
        origin_result=company_experiment_result,
        output_fig="results/comparison_zip_whiskers.png"
    )

    return results_list

def run_country_exp():
    config1 = Config(
        search=SearchConfig(iterations=20),
        data=DataConfig(
            directory_path='data/country_extend_1/seller',
            buyer_csv='data/country_extend_1/buyer/buyer_gini.csv',
            join_keys=[['country']],
            target_feature='value',
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

    return results_list


def run_country_alite_exp():
    config1 = Config(
        search=SearchConfig(iterations=20),
        data=DataConfig(
            directory_path='data/alite_only_searched_buyer_gini.csv_country_with_origin_datasets/seller',
            buyer_csv='data/alite_only_searched_buyer_gini.csv_country_with_origin_datasets/buyer/buyer_gini.csv',
            join_keys=[['country']],
            target_feature='value',
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

    return results_list

def run_ticker_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/stock_ticker_datasets/seller',
            buyer_csv='data/stock_ticker_datasets/buyer/financial data sp500 companies.csv',
            join_keys=[['Ticker']],
            target_feature='Income Before Tax',
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
    config2 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/alite_searched_stock_ticker_datasets/seller',
            buyer_csv='data/alite_searched_stock_ticker_datasets/buyer/financial data sp500 companies.csv',
            join_keys=[['Ticker']],
            target_feature='Income Before Tax',
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
    company_2 = ScaledExperiment(config2)
    
    company_experiment_result = company.run()
    company_experiment_result_2 = company_2.run()

    # Return as list
    results_list = [company_experiment_result]
    results_list.append(company_experiment_result_2)

    # plot
    plot_two_experiments_results_max_kitana_alite(
        origin_result=company_experiment_result,
        alite_result=company_experiment_result_2,
        output_fig="results/comparison_ticker_alite_whiskers.png"
    )

    return results_list



def run_ticker_alite_oriented_exp():
    config1 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/alite_only_searched_stock_ticker_kitana_origin_datasets/seller',
            buyer_csv='data/alite_only_searched_stock_ticker_kitana_origin_datasets/buyer/financial data sp500 companies.csv',
            join_keys=[['Ticker']],
            target_feature='Income Before Tax',
            one_target_feature=False,
            need_to_clean_data=True
        ),
        experiment=ExperimentConfig(
            plot_results=True,
            results_dir='results/'
        ),
        logging=LoggingConfig(
            level='CRITICAL',
            file='logs/experiment.log'
        )
    )
    config2 = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path='data/alite_only_searched_stock_ticker_datasets/seller',
            buyer_csv='data/alite_only_searched_stock_ticker_datasets/buyer/financial data sp500 companies.csv',
            join_keys=[['Ticker']],
            target_feature='Income Before Tax',
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
    company_2 = ScaledExperiment(config2)
    
    company_experiment_result = company.run()
    company_experiment_result_2 = company_2.run()

    # Return as list
    results_list = [company_experiment_result]
    results_list.append(company_experiment_result_2)

    # plot
    plot_two_experiments_results_max_kitana_alite(
        origin_result=company_experiment_result,
        alite_result=company_experiment_result_2,
        output_fig="results/ticker_kitana_alite_oriented_exp.png"
    )

    return results_list


if __name__ == "__main__":
    exp_function = run_country_exp
    result = exp_function()
    with open(f"results/{exp_function.__name__}.json", "w") as f:
        json.dump(result, f)
    # parser = argparse.ArgumentParser(description="Run specific experiment function.")
    # parser.add_argument("--exp", type=str, required=True, help="Name of the experiment to run.")
    # args = parser.parse_args()
    
    # exp_mapping = {
    #     "run_company_exp": run_company_exp,
    #     "run_university_exp": run_university_exp,
    #     "run_fifa_exp": run_fifa_exp,
    #     "run_book_exp": run_book_exp,
    #     "run_currency_exp": run_currency_exp,
    #     "run_movie_exp": run_movie_exp,
    #     "run_zip_exp": run_zip_exp
    # }

    # if args.exp in exp_mapping:
    #     final_results = exp_mapping[args.exp]()
    #     for i, result in enumerate(final_results):
    #         with open(f"results/{args.exp}_{i}.json", "w") as f:
    #             json.dump(result, f)
    # else:
    #     print("Invalid experiment name provided")
