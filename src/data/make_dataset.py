# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# functions for preprocessing
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # LOAD RAW DATA
    raw_data = pd.read_csv(input_filepath)

    # REMOVE OUTLIERS
    # use IQR method to get mask of outliers for each non-Boolean column
    outlier_masks = []

    for data_col in ['IT Spend', 'Employee Count', 'PC Count', 'Size', 'Revenue']:
        q1 = np.percentile(raw_data[data_col], 25)
        q3 = np.percentile(raw_data[data_col], 75)
        iqr = q3 - q1

        outlier_masks.append(((raw_data[data_col] > q3 + 1.5*iqr) | (raw_data[data_col] < q1 - 1.5*iqr)))

    # combine outlier masks; create inverted mask to indicate non-outliers
    mask = [any(tup) for tup in zip(*outlier_masks)]
    inverted_mask = np.invert(mask)

    # apply inverted mask to clean data (to keep only non-outliers)
    clean_data = raw_data[inverted_mask].copy()

    # STORE CLEAN DATA
    clean_data.to_csv(output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
