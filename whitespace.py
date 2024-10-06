import pandas as pd


def find_whitespace_cells(df: pd.DataFrame):
    # Initialise an empty list to store the cell locations
    whitespace_cells = []

    # Loop through each cell in the DataFrame
    for row_idx, row in df.iterrows():
        for col_idx, value in row.items():
            # Check if the value is a string and if it has leading or trailing whitespace
            if isinstance(value, str) and (value != value.strip()):
                whitespace_cells.append((row_idx, col_idx, value))

    return whitespace_cells


# Example usage:
data = {
    "col1": ["  leading", "no_whitespace", "trailing  "],
    "col2": ["no_issue", " leading_space ", " clean "],
    "col3": [1, 2, 3],  # Non-string column
}


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # Use map to apply strip() to each cell if it's a string
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)


def trim_whitespace_optimised(df: pd.DataFrame) -> pd.DataFrame:
    # Only apply to columns with object (string) data types
    string_columns = df.select_dtypes(include=["object", "string"])
    df[string_columns.columns] = string_columns.apply(lambda col: col.str.strip())

    return df


df = pd.DataFrame(data)
print(df)
whitespace_cells = find_whitespace_cells(df)
print(whitespace_cells)


cleaned_df = trim_whitespace_optimised(df)
print(cleaned_df)

whitespace_cells = find_whitespace_cells(cleaned_df)
print(whitespace_cells)
