#Import libraries
import pandas as pd
import numpy as np
import re
import random
from io import StringIO


#Define preprocessing class
class Preprocessor():

    def __init__(self, question):

        self.question = question

    def parse_question(self):
        """
        Divide question into:
        - drive test dataframe
        - engineering dataframe
        """

        # Split sections
        parts = self.question.split(
            "Engeneering parameters data as follows："
        )

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
            eng_df[col] = pd.to_numeric(
                eng_df[col],
                errors="coerce"
            )
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

