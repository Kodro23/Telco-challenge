#Import libraries
import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.preprocessing import LabelEncoder


#Define preprocessing class
class Preprocessor():

    def __init__(self, question):

        self.question = question.replace("：", ":").replace("\u00A0", " ").replace("\r\n", "\n")

    def parse_question(self):
        """
        Divide question into:
        - drive test dataframe
        - engineering dataframe
        """
        # Split sections
        parts = re.split(r"\s*Eng[a-z]*\s+parameters\s+data\s+as\s+follows\s*:\s*",self.question,flags=re.IGNORECASE)
        if len(parts) < 2:
            raise ValueError("Engineering section not found")

        drive_part = parts[0]
        eng_part = parts[1]

        # -------------------------
        # DRIVE TEST TABLE
        # -------------------------

        drive_lines = []
        started = False

        for line in drive_part.split("\n"):

            if "Timestamp|" in line:
                started = True

            if started and "|" in line:
                drive_lines.append(line)

        drive_text = "\n".join(drive_lines)

        drive_df = pd.read_csv(
            StringIO(drive_text),
            sep="|"
        )
        drive_df.columns = drive_df.columns.str.strip()
        # replace "-" with NaN
        drive_df = drive_df.replace("-", np.nan)
        for col in [c for c in drive_df.columns if c!="Timestamp"]:
            drive_df[col] = pd.to_numeric(
                drive_df[col],
                errors="coerce"
            )
        # -------------------------
        # ENGINEERING TABLE
        # -------------------------

        eng_lines = []

        started = False

        for line in eng_part.split("\n"):

            if "gNodeB ID|" in line:
                started = True

            if started and "|" in line:
                eng_lines.append(line)

        eng_text = "\n".join(eng_lines)

        eng_df = pd.read_csv(
            StringIO(eng_text),
            sep="|"
        )
        eng_df.columns = eng_df.columns.str.strip()
        # replace "-" with NaN
        eng_df = eng_df.replace("-", np.nan)
        for col in ["Cell ID", "Longitude", "Latitude", "Mechanical Azimuth", "Mechanical Downtilt", "Digital Tilt", "Digital Azimuth", "Height", "PCI", "Max Transmit Power"]:
            try:
                eng_df[col] = pd.to_numeric(
                    eng_df[col],
                    errors="coerce"
                )
            except:
                pass
        return drive_df, eng_df

    def merge_features(self, drive_df, eng_df):
        """
        Merge categories of features into one dataframe
        """

        merged = drive_df.merge(
            eng_df,
            left_on="5G KPI PCell RF Serving PCI",
            right_on="PCI",
            how="left"
        )

        return merged

    def build_sequence(self):

        """
        Transform the dataframes into one row by telelog id
        """

        drive_df, eng_df = self.parse_question()

        merged = self.merge_features(
            drive_df,
            eng_df
        )
        merged.columns = merged.columns.str.strip()
       
        # convert numeric
        merged["Timestamp"] = pd.to_datetime(merged["Timestamp"])
        return merged

def encode_column(df, col, encoders=None, training=False):
    """
    Label encode if encoders not already fixed
    """
    df[col] = df[col].astype(str)
    if training:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    else:
        le = encoders[col]
        df[col] = le.transform(df[col])
    return df

class FeatureBuilder:
    def __init__(self, merged_df, encoders=None, training=False):
        self.df = merged_df
    def build(self):

        df = self.df.copy()

        # fix duplicates
        df = df.rename(columns={
            "Longitude_x": "Longitude",
            "Latitude_x": "Latitude",
            "Longitude_y": "cell_Longitude",
            "Latitude_y": "cell_Latitude"
        })

        # sort FIRST
        df = df.sort_values("Timestamp")

        # drop useless columns
        df = df.drop(columns=["Timestamp","Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI",
                                "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI", 
                                "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI",
                                "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]",    
                                "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 Filtered Tx BRSRP [dBm]",    
                                "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 Filtered Tx BRSRP [dBm]","gNodeB ID","Cell ID"])

        # handle missing values
        ids = df["ID"]
        df = df.groupby("ID").ffill().bfill()
        df["ID"] = ids
        df = df.ffill().bfill()
        return df

        
    