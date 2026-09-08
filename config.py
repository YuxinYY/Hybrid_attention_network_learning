import argparse

argparser = argparse.ArgumentParser()
#data is not published, you may use your own datasets

argparser.add_argument(
    '--submissionandcomments_dir',
    type=str,
    default='data_processing/2023_sample.csv'

    # required=True,
    # help='Path to your private text data (not included due to license)'
)

argparser.add_argument(
    '--sp_stocks_dir',
    type=str,
    default='matched_stocks.csv'
)

argparser.add_argument(
    '--stocks',
    type=str,
    default='stocks.csv'
)

argparser.add_argument(
    '--daily_return_nyse',
    type=str,
    default='./nonuploads/daily_nyse_return.csv'
)

argparser.add_argument(
    '--eps_surprise_dir',
    type=str,
    default='training/datasets/earningreleasedatesandsurprises.csv'
    # required=True,
    # help='Path to your private earnings surprice data (not included due to license). May refer to :https://www.alphavantage.co/documentation/.'
)

argparser.add_argument(#for saving our training and testing datasets
    '--dataset_save_dir',
    type=str,
    default='training/datasets/'
)



args, unknown = argparser.parse_known_args()
# import sys
# if "__file__" in globals():  # Only parse when run as script, not when imported in notebook
#     args = argparser.parse_args()
# else:
#     args = argparser.parse_args([])  # Parse nothing (prevents Jupyter crash)

