import pandas as pd
import numpy as np
import math

class data_preprocessor():
    def get_num_cols(self, df, join_keys):
        def is_numeric(val):
            # Check for NaN (pandas NaN or math NaN)
            if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
                return False
            # Check for numeric types including numpy types
            return isinstance(val, (int, float, complex, np.integer, np.floating)) and not isinstance(val, bool)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        num_cols = [col for col in df.columns if is_numeric(df[col].iloc[0])]
        display_cols = [col for col in df.columns]
        for col in df.columns:
            nan_fraction = df[col].apply(lambda x: x == '' or pd.isna(x)).mean()
            # print(nan_fraction, col)
            if nan_fraction > 0.4:
                display_cols.remove(col)

        # Check if the first row has any NaN values
        df.fillna(0, inplace=True)

        for col in num_cols[:]:  # Iterate over a copy of num_cols
            has_string = df[col].apply(lambda x: isinstance(x, str)).any()

            if has_string:
                # Calculate the fraction of non-numeric (including NaN) entries
                non_numeric_fraction = df[col].apply(lambda x: not is_numeric(x)).mean()

                if non_numeric_fraction > 0.5:
                    # Remove the column from num_cols if more than half entries are non-numeric
                    num_cols.remove(col)
                else:
                    # Replace non-numeric entries with NaN and then fill them with the mean
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].mean(), inplace=True)
        if join_keys.difference(set(list(df.columns))): 
            return [], None
        else:
            for ele in join_keys.difference(set(display_cols)):
                display_cols = [ele] + display_cols
        return num_cols
    