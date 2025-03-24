import os
import tkinter as tk
import pandas as pd
import numpy as np
import joblib
import re
import chardet

from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, messagebox
from regipy import RegistryHive
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.feature_selection import RFE


###########################################################################
# 1) CLEANING FUNCTION
###########################################################################
def clean_dataframe(df, drop_all_zero_rows=True):
    """
    Cleans a DataFrame by:
      - Converting non-printable or unreadable data to '0'
      - Optionally dropping rows that are *entirely* zeros
      - Does NOT drop columns (even if they are all zero)
    """

    def clean_text(value):
        # If NaN or empty => '0'
        if pd.isna(value) or value == "":
            return "0"

        # If it's raw bytes => decode or else '0'
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                return "0"

        # If string => strip out non-printable ASCII
        if isinstance(value, str):
            value = re.sub(r'[^\x20-\x7E]', '', value)
            return value.strip() if value.strip() else "0"

        # Otherwise try to stringify
        return str(value)

    # Clean each cell
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    # Replace any lingering blanks with '0'
    df.replace("", "0", inplace=True)
    df.fillna("0", inplace=True)

    # Optionally drop rows that are all '0'
    if drop_all_zero_rows:
        df = df[~df.eq("0").all(axis=1)]

    # Do NOT drop columns, even if they're all "0"
    return df

###########################################################################
# HELPER: SAFE CSV READER THAT SKIPS BAD LINES
###########################################################################
def safe_read_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file but skips any badly formed lines (which cause the
    'C error: Expected X fields ... saw Y' in pandas). Returns a DataFrame
    of all successfully-read rows.
    """
    import pandas as pd
    try:
        df = pd.read_csv(
            filepath,
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'  # Skip lines that don't match expected columns
        )
        return df
    except Exception as e:
        print(f"[safe_read_csv] Error reading '{filepath}': {e}")
        return pd.DataFrame()

###########################################################################
# HELPER: MERGE DUPLICATE COLUMNS
###########################################################################
def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the CSV has duplicate column names, merge them. Here, we simply
    'combine' by taking first non-NA values from left to right.
    """
    from collections import defaultdict
    
    # Check if there are any duplicates first:
    duplicates_exist = df.columns.duplicated().any()
    if not duplicates_exist:
        # No duplicate column names, just return
        return df
    
    # We rebuild a new DataFrame by grouping all columns with the same name
    new_df = pd.DataFrame()
    unique_cols = df.columns.unique()
    
    for col in unique_cols:
        # Extract all columns with this same name
        same_name_slice = df.loc[:, df.columns == col]
        
        # Example strategy: forward/backward fill across these columns, then take one series
        merged_series = same_name_slice.bfill(axis=1).ffill(axis=1).iloc[:, 0]
        
        new_df[col] = merged_series
    
    return new_df

###########################################################################
# HELPER: REMOVE NON-TRAINING COLUMNS
###########################################################################
def remove_non_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are certainly not used for training.
    """
    needed_columns = {
    # --- Raw / Original columns referenced ---
    "Key",
    "Label",
    "Tactic",

    # --- Numeric columns from registry parsing ---
    "Depth",
    "Key Size",
    "Subkey Count",
    "Value Count",
    "Value Processed",

    # --- OHE columns for "Path Category" ---
    "PathCategory_Startup Path",
    "PathCategory_Service Path",
    "PathCategory_Network Path",
    "PathCategory_Other Path",

    # --- OHE columns for "Type Group" ---
    "TypeGroup_String",
    "TypeGroup_Numeric",
    "TypeGroup_Binary",
    "TypeGroup_Others",

    # --- OHE columns for "Key Name Category" ---
    "KeyNameCategory_Run Keys",
    "KeyNameCategory_Service Keys",
    "KeyNameCategory_Security and Configuration Keys",
    "KeyNameCategory_Internet and Network Keys",
    "KeyNameCategory_File Execution Keys",
    "KeyNameCategory_Other Keys",
    }
    
    # Make a list of columns to drop, i.e. those not in 'needed_columns'
    drop_cols = [c for c in df.columns if c not in needed_columns]
    
    if drop_cols:
        print(f"[remove_non_training_columns] Dropping columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return df

###########################################################################
# 2) CSV READER WITH FALLBACKS
###########################################################################
def read_csv_with_fallbacks(filepath):
    """
    Reads a CSV file with encoding detection and applies data cleaning.
    - Detects encoding using chardet
    - Reads CSV while replacing errors
    - Cleans the dataframe (fill unparsable with '0')
    - Keeps columns even if they are all zero
    - Optionally drops rows that are 100% zero
    """
    try:
        # Detect encoding by reading a chunk
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read(10000))
            encoding = result['encoding']

        print(f"Detected encoding: {encoding}")
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            encoding_errors='replace',
            dtype=str,  # Force string columns
            low_memory=False
        )

        # Clean up data => fill unparsable cells with "0", etc.
        df = clean_dataframe(df, drop_all_zero_rows=True)
        return df

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()  # Return empty on error


###########################################################################
# 3) MAIN MILOTIC CLASS
###########################################################################
class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("750x700")

        # User-provided paths
        self.sHivePath = ''
        self.sMaliciousKeysPath = ''
        self.sTaggedKeysPath = ''
        self.sTrainingDatasetPath = ''
        self.sModelOutputDir = os.getcwd()
        self.sClassifyCsvPath = ''
        self.sRawParsedCsvPath = ''
        self.sLabelModelPath = ''
        self.sTacticModelPath = ''
        self.sPersistenceModelPath = ''

        # Will be set after RFE in trainAndEvaluateModels
        self.selected_features = None

        self.setupUI()

    ###########################################################################
    #                           GUI Setup
    ###########################################################################
    def setupUI(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Row 0
        ttk.Label(frame, text="Hive Path:").grid(row=0, column=0, sticky='e')
        self.hivePathInput = ttk.Entry(frame, width=50)
        self.hivePathInput.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Set Hive Path", command=self.setHivePath).grid(row=0, column=2, padx=5)

        # Row 1
        ttk.Label(frame, text="Malicious Keys File:").grid(row=1, column=0, sticky='e')
        self.maliciousKeysInput = ttk.Entry(frame, width=50)
        self.maliciousKeysInput.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Set Malicious Keys", command=self.setMaliciousKeysPath).grid(row=1, column=2, padx=5)

        # Row 2
        ttk.Label(frame, text="Tagged Keys File:").grid(row=2, column=0, sticky='e')
        self.taggedKeysInput = ttk.Entry(frame, width=50)
        self.taggedKeysInput.grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Set Tagged Keys", command=self.setTaggedKeysPath).grid(row=2, column=2, padx=5)

        # Row 3
        ttk.Label(frame, text="Training Dataset (Optional):").grid(row=3, column=0, sticky='e')
        self.trainingDatasetInput = ttk.Entry(frame, width=50)
        self.trainingDatasetInput.grid(row=3, column=1, padx=5)
        ttk.Button(frame, text="Set Training Dataset", command=self.setTrainingDatasetPath).grid(row=3, column=2, padx=5)

        # Row 4
        ttk.Label(frame, text="Raw Parsed CSV (Optional):").grid(row=4, column=0, sticky='e')
        self.rawParsedCsvInput = ttk.Entry(frame, width=50)
        self.rawParsedCsvInput.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Set Raw Parsed CSV", command=self.setRawParsedCsvPath).grid(row=4, column=2, padx=5)

        # Row 5
        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=5, column=0, sticky='e')
        self.classifyCsvInput = ttk.Entry(frame, width=50)
        self.classifyCsvInput.grid(row=5, column=1, padx=5)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=5, column=2, padx=5)

        # Row 6
        ttk.Label(frame, text="Label Model (Optional):").grid(row=6, column=0, sticky='e')
        self.labelModelInput = ttk.Entry(frame, width=50)
        self.labelModelInput.grid(row=6, column=1, padx=5)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=6, column=2, padx=5)

        # Row 7
        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=7, column=0, sticky='e')
        self.tacticModelInput = ttk.Entry(frame, width=50)
        self.tacticModelInput.grid(row=7, column=1, padx=5)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=7, column=2, padx=5)

        # Row 8
        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=8, column=0, sticky='e')
        self.persistenceModelInput = ttk.Entry(frame, width=50)
        self.persistenceModelInput.grid(row=8, column=1, padx=5)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=8, column=2, padx=5)

        # Row 9/10: Buttons
        ttk.Button(frame, text="Make Dataset", command=self.makeDataset).grid(row=9, column=1, pady=10)
        ttk.Button(frame, text="Start ML Process", command=self.executeMLProcess).grid(row=10, column=1, pady=10)

        # Row 11: Metrics
        self.metricsList = ttk.Treeview(frame, columns=("Metric", "Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value", text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value", width=500, anchor="w")
        self.metricsList.grid(row=11, column=0, columnspan=3, pady=10)

    ###########################################################################
    #                            Path Setters
    ###########################################################################
    def setHivePath(self):
        self.sHivePath = self.hivePathInput.get().strip()
        messagebox.showinfo("Path Set", f"Hive path set to: {self.sHivePath}")

    def setMaliciousKeysPath(self):
        self.sMaliciousKeysPath = self.maliciousKeysInput.get().strip()
        messagebox.showinfo("Path Set", f"Malicious keys file set to: {self.sMaliciousKeysPath}")

    def setTaggedKeysPath(self):
        self.sTaggedKeysPath = self.taggedKeysInput.get().strip()
        messagebox.showinfo("Path Set", f"Tagged keys file set to: {self.sTaggedKeysPath}")

    def setTrainingDatasetPath(self):
        self.sTrainingDatasetPath = self.trainingDatasetInput.get().strip()
        messagebox.showinfo("Path Set", f"Training dataset path set to: {self.sTrainingDatasetPath}")

    def setRawParsedCsvPath(self):
        self.sRawParsedCsvPath = self.rawParsedCsvInput.get().strip()
        messagebox.showinfo("Path Set", f"Raw parsed CSV set to: {self.sRawParsedCsvPath}")

    def setClassifyCsvPath(self):
        self.sClassifyCsvPath = self.classifyCsvInput.get().strip()
        messagebox.showinfo("Path Set", f"CSV to classify set to: {self.sClassifyCsvPath}")

    def setLabelModelPath(self):
        self.sLabelModelPath = self.labelModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Label model set to: {self.sLabelModelPath}")

    def setTacticModelPath(self):
        self.sTacticModelPath = self.tacticModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Tactic model set to: {self.sTacticModelPath}")

    def setPersistenceModelPath(self):
        self.sPersistenceModelPath = self.persistenceModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Persistence model set to: {self.sPersistenceModelPath}")

    ###########################################################################
    #                          MAKE DATASET
    ###########################################################################
    def parseRegistry(self, hive_path):
        """
        Parse the registry hive into raw columns: 'Key','Depth', 'Name','Value','Type', etc.
        Then clean the resulting DataFrame with clean_dataframe()
        """
        xData = []
        subkey_counts = {}
        try:
            with ThreadPoolExecutor() as executor:
                hive = RegistryHive(hive_path)
                for subkey in hive.recurse_subkeys():
                    sKeyPath = subkey.path
                    parent_path = '\\'.join(sKeyPath.split('\\')[:-1])
                    subkey_counts[parent_path] = subkey_counts.get(parent_path, 0) + 1

                    d = sKeyPath.count('\\')
                    ksz = len(sKeyPath.encode('utf-8'))
                    vcount = len(subkey.values)
                    scount = subkey_counts.get(sKeyPath, 0)

                    for val in subkey.values:
                        try:
                            value_str = str(val.value) if val.value else "0"
                        except Exception:
                            value_str = "0"

                        try:
                            name_str = str(val.name) if val.name else "0"
                        except Exception:
                            name_str = "0"

                        try:
                            type_str = str(val.value_type) if val.value_type else "0"
                        except Exception:
                            type_str = "0"

                        xData.append({
                            "Key": sKeyPath if sKeyPath else "0",
                            "Depth": d if d is not None else "0",
                            "Key Size": ksz if ksz is not None else "0",
                            "Subkey Count": scount if scount is not None else "0",
                            "Value Count": vcount if vcount is not None else "0",
                            "Name": name_str,
                            "Value": value_str,
                            "Type": type_str
                        })

            df = pd.DataFrame(xData)
            # Clean the DataFrame => fill unparseable with '0'
            df = clean_dataframe(df, drop_all_zero_rows=True)
            return df

        except Exception as e:
            print(f"Error parsing hive: {e}")
            return pd.DataFrame()


    def makeDataset(self):
        """
        Parses registry data, applies labels, and prepares the dataset.
        - Cleans data and sets unparsable values to '0'
        - Keeps columns even if they're all zero
        - If any row is entirely '0', we drop that row
        - Then saves/updates the training CSV
        """
        try:
            if not os.path.exists(self.sHivePath):
                raise FileNotFoundError("Hive path not found.")

            print("Parsing registry data...")
            df_raw = self.parseRegistry(self.sHivePath)

            print("Applying labels...")
            df_labeled = self.applyLabels(df_raw)

            # Save or append raw-labeled data
            if self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                self.appendToExistingCsv(df_labeled, self.sRawParsedCsvPath)
                print(f"Appended raw-labeled data to: {self.sRawParsedCsvPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_raw_path = os.path.join(self.sModelOutputDir, f"raw_parsed_{ts}.csv")
                df_labeled.to_csv(new_raw_path, index=False)
                self.sRawParsedCsvPath = new_raw_path
                self.rawParsedCsvInput.delete(0, "end")
                self.rawParsedCsvInput.insert(0, new_raw_path)
                print(f"Created new raw-labeled CSV: {new_raw_path}")

            # Preprocess data for training
            print("Preprocessing for training dataset...")
            df_preproc = self.preprocessData(df_labeled)

            # Select training columns
            final_cols = self.selectTrainingColumns(df_preproc)
            df_preproc = df_preproc[final_cols]

            # Fill NaN or blank with '0'
            df_preproc.replace("", "0", inplace=True)
            df_preproc.fillna("0", inplace=True)

            # Save or append to training dataset
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                self.appendToExistingCsv(df_preproc, self.sTrainingDatasetPath)
                print(f"Appended to training dataset: {self.sTrainingDatasetPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_train_path = os.path.join(self.sModelOutputDir, f"training_dataset_{ts}.csv")
                df_preproc.to_csv(new_train_path, index=False)
                self.sTrainingDatasetPath = new_train_path
                self.trainingDatasetInput.delete(0, "end")
                self.trainingDatasetInput.insert(0, new_train_path)
                print(f"Created new training dataset: {new_train_path}")
                messagebox.showinfo("Dataset Created", f"New training dataset: {new_train_path}")

        except Exception as e:
            msg = f"Error in makeDataset: {e}"
            print(msg)
            messagebox.showerror("Error", msg)

    def selectTrainingColumns(self, df):
        """
        Keep only:
          - 'Key'
          - 'Label'
          - 'Tactic'
          - numeric columns like: Depth, Key Size, Subkey Count, Value Count, Value Processed
          - dummy columns that start with PathCategory_, TypeGroup_, KeyNameCategory_
        """
        keep_cols = []
        for c in df.columns:
            if c == 'Key':
                keep_cols.append(c)
            elif c in ('Label','Tactic'):
                keep_cols.append(c)
            elif c in ('Depth','Key Size','Subkey Count','Value Count','Value Processed'):
                keep_cols.append(c)
            elif c.startswith('PathCategory_') or c.startswith('TypeGroup_') or c.startswith('KeyNameCategory_'):
                keep_cols.append(c)
        # Only return columns that actually exist
        return [col for col in keep_cols if col in df.columns]

    def appendToExistingCsv(self, new_df: pd.DataFrame, csv_path: str):
        """
        Appends new_df to an existing CSV (csv_path), uniting columns and matching
        column order. Missing columns/cells are filled with '0', so there are no empty
        rows/columns in the final CSV.

        Steps:
          1) Read the existing CSV (if it exists).
          2) Use new_df's columns as the 'reference order'.
          3) If the existing CSV has extra columns, append them to the reference.
          4) Reindex both DataFrames to that unified column list, in that order.
          5) Fill missing cells with '0'.
          6) Concatenate row-wise and save.
        """
        try:
            # If the file doesn't exist yet, just save the new DataFrame
            if not csv_path or not os.path.exists(csv_path):
                new_df.to_csv(csv_path, index=False)
                print(f"[appendToExistingCsv] Created new CSV: {csv_path}")
                return

            # 1) Read existing CSV
            existing_df = pd.read_csv(csv_path, dtype=str)  # read as string

            # 2) new_df's columns = the "desired" order
            reference_cols = list(new_df.columns)

            # 3) If the existing CSV has extra columns, keep them too (append them at the end)
            for col in existing_df.columns:
                if col not in reference_cols:
                    reference_cols.append(col)

            # 4) Reindex both DataFrames to that unified list, in that order
            existing_df = existing_df.reindex(columns=reference_cols)
            new_df = new_df.reindex(columns=reference_cols)

            # 5) Fill empty or NaN cells with '0'
            existing_df.fillna("0", inplace=True)
            new_df.fillna("0", inplace=True)
            existing_df.replace("", "0", inplace=True)
            new_df.replace("", "0", inplace=True)

            # 6) Concatenate row-wise
            combined = pd.concat([existing_df, new_df], ignore_index=True)

            # (Optional) Drop rows that are all '0' if you never want fully empty rows:
            # combined = combined.loc[~combined.eq("0").all(axis=1)]

            # Finally, save combined
            combined.to_csv(csv_path, index=False)
            print(f"[appendToExistingCsv] Appended + Reordered CSV: {csv_path}")

        except Exception as ex:
            print(f"Error appending to CSV ({csv_path}): {ex}")
            # As a fallback, just save new_df so data isn't lost
            new_df.to_csv(csv_path, index=False)

    ###########################################################################
    #                          APPLY LABELS
    ###########################################################################
    def applyLabels(self, df):
        """Add 'Label','Tactic' to raw. Keep 'Key'. """
        if 'Key' not in df.columns:
            raise KeyError("No 'Key' column in data for labeling.")

        malicious_entries = []
        if self.sMaliciousKeysPath and os.path.exists(self.sMaliciousKeysPath):
            with open(self.sMaliciousKeysPath,'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in re.split(r'[\|;]', line.strip()) if p.strip()]
                    entry = {
                        "Key": re.sub(r'\\+', r'\\', parts[0]) if len(parts)>0 else None,
                        "Name": parts[1].strip() if len(parts)>1 else None,
                        "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts)>2 else None,
                        "Type": parts[3].strip() if len(parts)>3 else None
                    }
                    malicious_entries.append(entry)

        tagged_entries = []
        if self.sTaggedKeysPath and os.path.exists(self.sTaggedKeysPath):
            with open(self.sTaggedKeysPath,'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in re.split(r'[\,\|;]', line.strip()) if p.strip()]
                    entry = {
                        "Key": re.sub(r'\\+', r'\\', parts[0]) if len(parts)>0 else None,
                        "Name": parts[1].strip() if len(parts)>1 else None,
                        "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts)>2 else None,
                        "Type": parts[3].strip() if len(parts)>3 else None,
                        "Tactic": parts[4].strip() if len(parts)>4 else "Persistence"
                    }
                    tagged_entries.append(entry)

        def is_malicious(row):
            row_key = re.sub(r'\\+', r'\\', str(row['Key']).lower())
            row_name = str(row['Name']).lower().strip()
            row_value = re.sub(r'\\+', r'\\', str(row['Value']).lower().strip())
            row_type = str(row['Type']).lower().strip()
            for e in malicious_entries:
                ekey_last = e['Key'].strip().split('\\')[-1].lower() if e['Key'] else ""
                row_key_last = row_key.split('\\')[-1]
                if row_key_last!=ekey_last:
                    continue
                if e['Name'] and row_name!=e['Name'].lower():
                    continue
                if e['Value'] and row_value!= e['Value'].lower():
                    continue
                if e['Type'] and row_type!= e['Type'].lower():
                    continue
                return 'Malicious'
            return 'Benign'

        def assign_tactic(row):
            row_key = re.sub(r'\\+', r'\\', str(row['Key']).lower())
            row_name = str(row['Name']).lower().strip()
            row_value = re.sub(r'\\+', r'\\', str(row['Value']).lower().strip())
            row_type = str(row['Type']).lower().strip()
            for e in tagged_entries:
                ekey_last = e['Key'].strip().split('\\')[-1].lower() if e['Key'] else ""
                row_key_last = row_key.split('\\')[-1]
                if row_key_last!=ekey_last:
                    continue
                if e['Name'] and row_name!=e['Name'].lower():
                    continue
                if e['Value'] and row_value!= e['Value'].lower():
                    continue
                if e['Type'] and row_type!= e['Type'].lower():
                    continue
                return e['Tactic']
            return 'None'

        df['Label'] = df.apply(is_malicious, axis=1)
        df['Tactic'] = df.apply(assign_tactic, axis=1)
        return df

    ###########################################################################
    #                         PREPROCESS (NO RFE)
    ###########################################################################
    def preprocessData(self, df):
        """
        Scale numeric, get dummies, keep 'Key' etc,
        do not do RFE here.
        """
        if df.empty:
            print("No data to preprocess.")
            return pd.DataFrame()

        xDf = df.copy()
        # For any numeric columns, fill NaN with mean => (already '0' in practice)
        numeric_df = xDf.select_dtypes(include=[np.number])
        xDf.fillna(numeric_df.mean(), inplace=True)

        # Path cat
        xDf['Path Category'] = xDf['Key'].apply(self.categorizePath)
        path_enc = pd.get_dummies(xDf['Path Category'], prefix='PathCategory')
        xDf = pd.concat([xDf, path_enc], axis=1)

        # Type group
        xDf['Type Group'] = xDf['Type'].apply(self.mapType)
        type_enc = pd.get_dummies(xDf['Type Group'], prefix='TypeGroup')
        xDf = pd.concat([xDf, type_enc], axis=1)

        # Key name cat
        xDf['Key Name Category'] = xDf['Name'].apply(self.categorizeKeyName)
        name_enc = pd.get_dummies(xDf['Key Name Category'], prefix='KeyNameCategory')
        xDf = pd.concat([xDf, name_enc], axis=1)

        # value processed
        xDf['Value Processed'] = xDf['Value'].apply(self.preprocessValue)

        # scale numeric
        minmax = MinMaxScaler()
        minmax_cols = ['Depth','Value Count','Value Processed']
        for col in minmax_cols:
            if col in xDf.columns:
                xDf[[col]] = minmax.fit_transform(xDf[[col]])

        robust = RobustScaler()
        robust_cols = ['Key Size','Subkey Count']
        for col in robust_cols:
            if col in xDf.columns:
                xDf[[col]] = robust.fit_transform(xDf[[col]])

        return xDf

    def categorizePath(self, p):
        if "Run" in p:
            return "Startup Path"
        elif "Services" in p:
            return "Service Path"
        elif "Internet Settings" in p:
            return "Network Path"
        return "Other Path"

    def mapType(self, t):
        type_map = {
            "String": ["REG_SZ","REG_EXPAND_SZ","REG_MULTI_SZ"],
            "Numeric": ["REG_DWORD","REG_QWORD"],
            "Binary": ["REG_BINARY"],
            "Others": ["REG_NONE","REG_LINK","0"]  # '0' -> unrecognized => Others
        }
        for g, vals in type_map.items():
            if t in vals:
                return g
        return "Others"

    def categorizeKeyName(self, kn):
        categories = {
            "Run Keys": ["Run","RunOnce","RunServices"],
            "Service Keys": ["ImageFileExecutionOptions","AppInit_DLLs"],
            "Security and Configuration Keys": ["Policies","Explorer"],
            "Internet and Network Keys": ["ProxyEnable","ProxyServer"],
            "File Execution Keys": ["ShellExecuteHooks"]
        }
        # Convert to lower or do partial checks
        low = kn.lower()
        for cat, keys in categories.items():
            if any(k.lower() in low for k in keys):
                return cat
        return "Other Keys"

    def preprocessValue(self, v):
        if isinstance(v,str):
            return len(v)
        return v

    ###########################################################################
    #            EXECUTE ML PROCESS (RFE here, not in makeDataset)
    ###########################################################################
    def executeMLProcess(self):
        """
        If no classify CSV is provided, we just use the training dataset for a
        test classification. Then we read the training CSV in a robust manner,
        fix duplicates, drop unneeded columns, and proceed with training.
        """
        try:
            # If user didn't provide a CSV to classify, just use the training set
            if not self.sClassifyCsvPath:
                print("No classify CSV provided, using training dataset for classification test.")
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            # -- 1) Read training CSV in a safer way (skip malformed lines)
            df_train = safe_read_csv(self.sTrainingDatasetPath)
            
            # -- 2) Merge duplicated column names if any
            df_train = merge_duplicate_columns(df_train)

            # -- 3) Drop columns that are truly not needed for training
            df_train = remove_non_training_columns(df_train)

            # -- 4) Clean up data, dropping rows that are entirely "0"
            df_train = clean_dataframe(df_train, drop_all_zero_rows=True)

            # -- 5) Proceed with training
            self.trainAndEvaluateModels(df_train)

            # -- 6) Classify the user-provided CSV (or the training set if none)
            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "Finished training & classification!")

        except Exception as ex:
            messagebox.showerror("Error", f"Error in ML process: {ex}")
    ###########################################################################
    #                TRAIN AND EVALUATE MODELS
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        3 RandomForest models: label, defense, persistence
        If single-class => forcibly flip ~30% to class '1' for demonstration
        RFE is done here only.
        """
        try:
            if df.empty:
                raise ValueError("Training dataset is empty")

            # Convert numeric columns
            for c in ['Depth','Key Size','Subkey Count','Value Count','Value Processed']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            # Remove columns not used for features
            exclude_cols = []
            for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                if c in df.columns:
                    exclude_cols.append(c)
            X_all = df.drop(columns=exclude_cols, errors='ignore').copy()

            # Convert leftover columns to numeric if possible
            for col in X_all.columns:
                if X_all[col].dtype == object:
                    # If they're "0" or other ints => parse them
                    X_all[col] = pd.to_numeric(X_all[col], errors='coerce').fillna(0)

            if 'Label' not in df.columns or 'Tactic' not in df.columns:
                raise ValueError("Missing 'Label'/'Tactic' in dataset")

            y_label = (df['Label']=='Malicious').astype(int)
            y_defense = (df['Tactic']=='Defense Evasion').astype(int)
            y_persist = (df['Tactic']=='Persistence').astype(int)

            # Force multi-class if single
            if y_label.nunique()<2:
                idx = np.random.choice(y_label.index, size=int(len(y_label)*0.3), replace=False)
                y_label.iloc[idx]=1
                print("Forced ~30% malicious for label model test.")
            if y_defense.nunique()<2:
                idx = np.random.choice(y_defense.index, size=int(len(y_defense)*0.3), replace=False)
                y_defense.iloc[idx]=1
                print("Forced ~30% 'Defense Evasion'.")
            if y_persist.nunique()<2:
                idx = np.random.choice(y_persist.index, size=int(len(y_persist)*0.3), replace=False)
                y_persist.iloc[idx]=1
                print("Forced ~30% 'Persistence'.")

            # RFE (with label) => pick top 10
            print("Performing RFE for feature selection with label as target.")
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=base_model, n_features_to_select=10)
            rfe.fit(X_all, y_label)
            self.selected_features = X_all.columns[rfe.support_]
            print("Selected features =>", list(self.selected_features))

            def grid_search_rf(Xp, yp):
                param_grid = {
                    'n_estimators':[50,100],
                    'max_depth':[None,10],
                    'min_samples_split':[2,5],
                    'min_samples_leaf':[1,2],
                    'bootstrap':[True,False]
                }
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                model = RandomForestClassifier(random_state=42)
                gs = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
                gs.fit(Xp, yp)
                return gs.best_estimator_

            def evaluate_model(model, Xd, yd, label_name=""):
                X_tr, X_te, y_tr, y_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                try:
                    y_scores = model.predict_proba(X_te)[:,1]
                except AttributeError:
                    y_scores = None

                acc = accuracy_score(y_te, y_pred)
                prec = precision_score(y_te, y_pred, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)
                f1v = f1_score(y_te, y_pred, zero_division=0)
                aucv = 0.0
                if y_scores is not None and len(np.unique(y_te))>1:
                    aucv = roc_auc_score(y_te, y_scores)

                print(f"{label_name} => Acc={acc:.4f},Prec={prec:.4f},Rec={rec:.4f},F1={f1v:.4f},AUC={aucv:.4f}")
                return {
                    f"{label_name} Accuracy": acc,
                    f"{label_name} Precision": prec,
                    f"{label_name} Recall": rec,
                    f"{label_name} F1": f1v,
                    f"{label_name} AUC": aucv
                }

            X_sel = X_all[self.selected_features].copy()

            # Label model
            print("Training Label Model...")
            label_model = grid_search_rf(X_sel, y_label)
            label_metrics = evaluate_model(label_model, X_sel, y_label, "Label Model")
            label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
            joblib.dump(label_model, label_model_path)
            self.sLabelModelPath = label_model_path

            # Defense
            print("Training Defense Evasion Model...")
            defense_model = grid_search_rf(X_sel, y_defense)
            defense_metrics = evaluate_model(defense_model, X_sel, y_defense, "Defense Evasion Model")
            defense_model_path = os.path.join(self.sModelOutputDir, "defense_model.joblib")
            joblib.dump(defense_model, defense_model_path)
            self.sTacticModelPath = defense_model_path

            # Persistence
            print("Training Persistence Model...")
            persistence_model = grid_search_rf(X_sel, y_persist)
            persist_metrics = evaluate_model(persistence_model, X_sel, y_persist, "Persistence Model")
            persist_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
            joblib.dump(persistence_model, persist_model_path)
            self.sPersistenceModelPath = persist_model_path

            # Merge metrics
            combined = {}
            combined.update(label_metrics)
            combined.update(defense_metrics)
            combined.update(persist_metrics)
            out_metrics = {k: f"{v:.4f}" for k,v in combined.items()}
            self.updateMetricsDisplay(out_metrics)

        except Exception as ex:
            raise RuntimeError(f"Training error: {ex}")

    ###########################################################################
    #                    CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path):
        """
        Classifies a CSV dataset using pre-trained models.
        - Uses robust encoding detection
        - Ensures missing columns are handled properly
        - Saves classification results to a new CSV
        """
        try:
            df = read_csv_with_fallbacks(csv_path)
            if df.empty:
                raise ValueError("No data to classify (empty DataFrame).")

            if 'Key' not in df.columns:
                df['Key'] = "UNKNOWN"

            # Convert columns to numeric where possible
            for col in df.columns:
                # If it's "Depth," etc. => numeric
                if col not in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            if self.selected_features is None or len(self.selected_features) == 0:
                raise ValueError("No selected features found. Did you run training?")

            label_model = joblib.load(self.sLabelModelPath)
            defense_model = joblib.load(self.sTacticModelPath)
            persistence_model = joblib.load(self.sPersistenceModelPath)

            # Exclude non-feature columns
            exclude_cols = ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']
            X_all = df.drop(columns=exclude_cols, errors='ignore')

            # Use only selected features (set missing => 0)
            for col in self.selected_features:
                if col not in X_all.columns:
                    X_all[col] = 0
            X_all = X_all[self.selected_features].copy()

            # Convert to numeric
            for col in X_all.columns:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce').fillna(0)

            # Predict labels
            y_scores_label = label_model.predict_proba(X_all)[:, 1]
            y_pred_label = np.where(y_scores_label >= 0.5, 'Malicious', 'Benign')

            # Predict tactics
            y_scores_defense = defense_model.predict_proba(X_all)[:, 1]
            y_pred_defense = np.where(y_scores_defense >= 0.5, 'Defense Evasion', 'None')

            y_scores_persist = persistence_model.predict_proba(X_all)[:, 1]
            y_pred_persist = np.where(y_scores_persist >= 0.5, 'Persistence', 'None')

            df['Predicted Label'] = y_pred_label
            # If defense is triggered, it's "Defense Evasion," else check persistence
            df['Predicted Tactic'] = np.where(y_pred_defense == 'Defense Evasion',
                                              'Defense Evasion',
                                              y_pred_persist)

            # Save classified output
            out_path = os.path.join(self.sModelOutputDir, f"classified_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(out_path, index=False)
            print(f"Classified output saved to: {out_path}")
            messagebox.showinfo("Classification Complete", f"Classified output saved to: {out_path}")

        except Exception as ex:
            msg = f"Classification error: {ex}"
            print(msg)
            messagebox.showerror("Error", msg)

    ###########################################################################
    #                       METRICS
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, val in metrics.items():
            self.metricsList.insert("", "end", values=(metric, val))


###########################################################################
# MAIN
###########################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
