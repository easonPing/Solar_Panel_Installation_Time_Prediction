import pandas as pd
import numpy as np
import os
import re
import datetime
from sklearn.preprocessing import StandardScaler

def read_table (file_path, **kwargs):
    """
    Select pandas reader based on file extension.
    Require Python 3.10+
    Supports .xlsx, .xls, .csv. Extend as needed.
    :param file_path:
    :param kwargs:
    :return: reader based on file extension
    """

    ext = os.path.splitext(file_path)[-1].lower()

    match ext:
        case ".xlsx" | ".xls":
            return pd.read_excel(file_path, **kwargs)
        case ".csv":
            return pd.read_csv(file_path, **kwargs)
        case _:
            raise ValueError("File extension not recognized. Only Support xlsx, xls, and csv")

def clean_column_names(columns):
    """
    Replace line breaks, and collapse multiple spaces.
    :param columns:
    :return:
    """

    cleaned_columns = []

    for col in columns:
        name = col.strip().replace("\n", " ")
        name = re.sub(r"\s+", " ", name)
        cleaned_columns.append(name)

    return cleaned_columns


def load_data (file_path = None, debug_dir="./checkpoints", target_colum = None, verbose = True):
    """
    Universal data loader for solar panel data.
    Assume header is the first row.
    Supports Excel and CSV files, cleans columns, parses all features and the target,
    factorizes categorical variables, and returns scaled data.

    :param file_path:
    :param target_colum:
    :param verbose:
    :return:
    """

    os.makedirs(debug_dir, exist_ok=True)

    # Read the data table
    df = read_table(file_path)
    df.columns = clean_column_names(df.columns.tolist())
    if verbose:
        print("Raw columns:", df.columns.tolist())
        print(df.head(2))

    # Remove the empty rows/columns and unnamed fields
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]

    # Set excluded columns, and calculate target (total install time (excluding drive time))
    direct_time_col = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"
    drive_time_col = "Drive Time"
    days_col = "Total # of Days on Site"
    exclude = [
        "Project ID",
        "Total # of Days on Site",
        "Total # Hourly Employees on Site",
        "Estimated # of Salaried Employees on Site",
        "Estimated Salary Hours",
        "Estimated Total Direct Time",
        "Estimated Total # of People on Site",
        "Notes",
        direct_time_col,
        drive_time_col,
        days_col
    ]

    df["Total Install Time (Excluding Drive)"] = df.apply(
        #lambda r: parse_target(r, direct_time_col, drive_time_col, days_col), axis=1
    )
    target = "Total Install Time (Excluding Drive)"
    debug_df = df[[direct_time_col, drive_time_col, days_col, target]].copy()
    debug_df.to_excel(os.path.join(debug_dir, "target_processing_debug.xlsx"))
