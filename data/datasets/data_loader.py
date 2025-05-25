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

def clean_angle(value):
    """
    Parse roof angle or azimuth values from various formats such as '30°' or '30/25'.
    If value contains '/', takes the mean; otherwise attempts direct float conversion.
    Returns NaN on failure.
    """
    if pd.isna(value): return np.nan
    value = str(value).replace("°", "").strip()
    if "/" in value:
        try:
            return np.mean([float(v) for v in value.split("/")])
        except ValueError:
            return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan

def convert_time_to_minutes(text):
    """
    Convert duration formats (e.g., '2h 15m', '90mins', '01:30', '2hours')
    or pandas Timedelta/datetime objects into minutes.
    Returns None if unable to parse.
    """
    if pd.isna(text) or text in ("", "nan", "None"): return None
    if isinstance(text, pd.Timedelta): return text.total_seconds() / 60
    if isinstance(text, (datetime.datetime, datetime.time)):
        return text.hour * 60 + text.minute + text.second / 60
    text = str(text).strip().lower()

    # Support "01:30" or "1:30:15"
    m = re.match(r"^(\d+):(\d+)(?::(\d+))?$", text)
    if m:
        h, m_, s = (int(x) if x else 0 for x in m.groups())
        return h * 60 + m_ + s / 60

    # Support formats like "2h 15m", "2hr 15min", "2 hours 15 minutes", "3hrs", "2 hour"
    m = re.match(r"^(\d+)\s*(h|hr|hrs|hour|hours)(?:\s*(\d+)\s*(m|min|mins|minute|minutes))?$", text)
    if m:
        h = int(m.group(1))
        m_ = int(m.group(3)) if m.group(3) else 0
        return h * 60 + m_

    # Support "90min", "90mins", "90 minutes"
    m = re.match(r"^(\d+)\s*(m|min|mins|minute|minutes)$", text)
    if m:
        return int(m.group(1))

    # Pure digits
    if text.isdigit(): return int(text)
    return None

def parse_target(row, direct_time_col, drive_time_col, days_col):
    """
    Parse the total install duration (excluding drive time) in minutes.
    Formula: total_install_time = total_direct_time - 2 * drive_time * days_on_site
    """
    total_direct = row[direct_time_col]
    drive_time = row[drive_time_col] if drive_time_col in row else 0
    days = row[days_col] if days_col in row else 1

    # Convert all to minutes
    total_direct_min = convert_time_to_minutes(total_direct)
    drive_time_min = convert_time_to_minutes(drive_time)
    days_val = float(days) if not pd.isna(days) else 1

    if total_direct_min is None:
        return None
    install_time = total_direct_min - 2 * drive_time_min * days_val
    return install_time


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
        lambda r: parse_target(r, direct_time_col, drive_time_col, days_col), axis=1
    )
    target = "Total Install Time (Excluding Drive)"
    debug_df = df[[direct_time_col, drive_time_col, days_col, target]].copy()
    debug_df.to_excel(os.path.join(debug_dir, "target_processing_debug.xlsx"))

    #Clean angle columns
    for ang_col in ["Tilt", "Azimuth"]:
        if ang_col in df.columns:
            df[ang_col] = df[ang_col].apply(clean_angle)

    # Select features
    features = [c for c in df.columns if c not in exclude + [target]]

    # Normalize common binary fields
    binary_fields = ["Squirrel Screen", "Consumption Monitoring", "Reinforcements"]
    for field in binary_fields:
        if field in df.columns:
            df[field] = df[field].apply(lambda x: "Yes" if str(x).lower() in ["1", "yes", "true"] else "No")

    # One-hot encode all categorical features
    cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()
    category_options = {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols}
    df_onehot = pd.get_dummies(df[features], columns=cat_cols, dummy_na=False)

    # Fill missing numerical values with 0 (categorical handled by get_dummies)
    for col in df_onehot.columns:
        if df_onehot[col].dtype in ["float64", "int64"]:
            df_onehot[col] = pd.to_numeric(df_onehot[col], errors="coerce").fillna(0)

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(df_onehot)
    X_df = pd.DataFrame(X, columns=df_onehot.columns)
    y = df[target]

    # Save option list for each original categorical column (for future input UI)
    import json
    with open(os.path.join(debug_dir, "category_options.json"), "w", encoding="utf-8") as f:
        json.dump(category_options, f, ensure_ascii=False, indent=2)

    if verbose:
        print("One-hot encoded columns:", df_onehot.columns.tolist())
        print("Original category options for UI/inputs:", category_options)
        print("Data loading and preprocessing complete. Feature count:", len(X_df.columns))
        print(X_df.head(2))
        print("Sample targets: \n", y.head(2))
        print()
    return X_df, y, category_options

# Usage example:
# X_df, y, category_options = load_data("your_file.xlsx")