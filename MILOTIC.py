import os
import tkinter as tk
import pandas as pd, chardet
import numpy as np
import joblib
import re
import chardet

from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, messagebox
from regipy import RegistryHive
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.feature_selection import RFE

###########################################################################
#                           MAIN MILOTIC CLASS
###########################################################################
class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("750x780")
        self.root.resizable(False, False)
        
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

        # Row 9: Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=10)
        btn_make = ttk.Button(button_frame, text="Make Dataset", command=self.makeDataset)
        btn_start = ttk.Button(button_frame, text="Start ML Process", command=self.executeMLProcess)
        btn_make.pack(side="left", padx=5)
        btn_start.pack(side="left", padx=5)

        # Row 10: Metrics
        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=10, column=0, columnspan=3, pady=10, sticky='nsew')
        self.metricsList = ttk.Treeview(metrics_frame, columns=("Metric","Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value", text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value", width=500, anchor="w")
        self.metricsList.pack(side="left", fill="both", expand=True)
        m_scroll = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metricsList.yview)
        m_scroll.pack(side="right", fill="y")
        self.metricsList.configure(yscrollcommand=m_scroll.set)

        # Row 11: Features
        feature_frame = ttk.Frame(frame)
        feature_frame.grid(row=11, column=0, columnspan=3, pady=10, sticky='nsew')
        self.featureList = ttk.Treeview(feature_frame, columns=("Feature","Importance"), show="headings")
        self.featureList.heading("Feature", text="Feature")
        self.featureList.heading("Importance", text="Importance")
        self.featureList.column("Feature", width=200, anchor="w")
        self.featureList.column("Importance", width=500, anchor="w")
        self.featureList.pack(side="left", fill="both", expand=True)
        f_scroll = ttk.Scrollbar(feature_frame, orient="vertical", command=self.featureList.yview)
        f_scroll.pack(side="right", fill="y")
        self.featureList.configure(yscrollcommand=f_scroll.set)

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
    #                          UTILITY
    ###########################################################################
    ###########################################################################
#  Put these two helpers at class level (right under the other methods)
###########################################################################
    @staticmethod
    def read_txt(path, default_tac="Persistence"):
        import os, re
        recs = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    parts = [p.strip() for p in re.split(r"[|;,]", line) if p.strip()]
                    if not parts:
                        continue
                    rec = {"Key": parts[0].lower()}
                    if len(parts) > 1: rec["Name"]  = parts[1].lower()
                    if len(parts) > 2: rec["Value"] = parts[2].lower()
                    if len(parts) > 3: rec["Type"]  = parts[3].lower()
                    rec["Tactic"] = parts[4] if len(parts) > 4 else default_tac
                    recs.append(rec)
        return recs


    @staticmethod
    def strict_match(ent, rk, rn, rv, rt):
        """
        • Exact path match by default  
        • Allow trailing '*' in artefact path to act as “starts-with” wildcard  
        • Optional Name / Value / Type must also match if present
        """
        kp = ent["Key"]
        if kp.endswith("*"):
            # wildcard ⇒ prefix match
            if not rk.startswith(kp[:-1]):
                return False
        else:
            if rk != kp:
                return False   # FULL path must match now

        if ent.get("Name")  and rn != ent["Name"]:  return False
        if ent.get("Value") and rv != ent["Value"]: return False
        if ent.get("Type")  and rt != ent["Type"]:  return False
        return True

    def safe_read_csv(self, path):
        """
        Read a CSV file with encoding detection and fallback for bad lines.
        """
        import chardet
        import pandas as pd

        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")

        with open(path, "rb") as f:
            raw = f.read(10000)
            enc = chardet.detect(raw)["encoding"] or "utf-8"

        return pd.read_csv(path, encoding=enc, encoding_errors="replace", dtype=str, low_memory=False, on_bad_lines="skip")
       
    # def read_csv_with_fallbacks(path):
    #    """
    #    For classifyCSV
    #    """
    #    raw = open(path, "rb").read(10000)
    #    enc = chardet.detect(raw)["encoding"] or "utf-8"
    #    df = pd.read_csv(path, encoding=enc, encoding_errors="replace", dtype=str, low_memory=False)
    #    return self.clean_dataframe(df_raw, drop_all_zero_rows=True, preserve_labels=True)

    ###########################################################################
    #                          MAKE DATASET
    ###########################################################################
    def clean_dataframe(self, df: pd.DataFrame, drop_all_zero_rows: bool = True, preserve_labels: bool = True) -> pd.DataFrame:
        """
        Normalises all cells to printable ASCII; blanks->'0'.
        Drops a row only when every numeric is 0 and
          every string is '0' and it is not labelled / tagged.
        """

        def clean_text(val):
            if pd.isna(val) or val == "":
                return "0"
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace") or "0"
            if isinstance(val, str):
                return re.sub(r"[^\x20-\x7E]", "", val).strip() or "0"
            return str(val)

        # == basic cleaning ============================================
        for col in df.columns:
            df[col] = df[col].apply(clean_text)
        df.replace("", "0", inplace=True)
        df.fillna("0", inplace=True)

        # == optional row prune ========================================
        if drop_all_zero_rows:
            numeric_ok = df.select_dtypes(include=[np.number]).sum(axis=1) > 0
            string_ok  = df.select_dtypes(include=[object]).ne("0").any(axis=1)
            keep_mask  = numeric_ok | string_ok
            if preserve_labels:
                keep_mask |= df["Label"].str.lower().eq("malicious")
                if "Tactic" in df.columns:
                    keep_mask |= df["Tactic"].str.lower().ne("none")
            df = df.loc[keep_mask]

        return df
    
    def parseRegistry(self, hive_path):
        """
        Parse the registry hive into raw columns and clean the result.
        """

        xData, subkey_counts = [], {}
        hive = RegistryHive(hive_path)
        for subkey in hive.recurse_subkeys():
            path = subkey.path or "0"
            parent = "\\".join(path.split("\\")[:-1])
            subkey_counts[parent] = subkey_counts.get(parent, 0) + 1
            depth = path.count("\\")
            ksz   = len(path.encode("utf-8"))
            vcount = len(subkey.values)
            scount = subkey_counts.get(path, 0)

            if not subkey.values:
                xData.append({
                    "Key": path, "Depth": depth, "Key Size": ksz,
                    "Subkey Count": scount, "Value Count": vcount,
                    "Name": "0", "Value": "0", "Type": "0"
                })
            else:
                for val in subkey.values:
                    try:    val_str = str(val.value) if val.value else "0"
                    except: val_str = "0"
                    try:    name_str = str(val.name)  if val.name  else "0"
                    except: name_str = "0"
                    try:    type_str = str(val.value_type) if val.value_type else "0"
                    except: type_str = "0"
                    xData.append({
                        "Key": path, "Depth": depth, "Key Size": ksz,
                        "Subkey Count": scount, "Value Count": vcount,
                        "Name": name_str, "Value": val_str, "Type": type_str
                    })

        df = pd.DataFrame(xData)
        return self.clean_dataframe(df, drop_all_zero_rows=True, preserve_labels=True)
        
    def makeDataset(self):
        """
        If a hive is supplied -> parse hive -> label -> optionally append to the
        existing raw-parsed CSV (new data).
        If only a raw CSV is supplied -> reload + re-label -> overwrite
        the same CSV (so it never duplicates rows).

        Afterwards preprocesses the labelled data and updates/creates the
        training-dataset CSV.
        """
        try:
            # ------------------------------------------------------------
            # 1)  LOAD SOURCE DATA
            # ------------------------------------------------------------
            source_is_hive = bool(self.sHivePath and os.path.exists(self.sHivePath))

            if source_is_hive:
                print("Parsing registry hive...")
                df_raw = self.parseRegistry(self.sHivePath)
            elif self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                print("Loading raw parsed CSV...")
                df_raw = pd.read_csv(self.sRawParsedCsvPath, dtype=str, low_memory=False)
            else:
                raise FileNotFoundError("No valid Hive Path or Raw Parsed CSV provided.")

            df_raw = self.clean_dataframe(
                df_raw, drop_all_zero_rows=True, preserve_labels=True
            )

            # ------------------------------------------------------------
            # 2)  APPLY LABELS / TACTIC TAGS
            # ------------------------------------------------------------
            print("Applying labels...")
            df_labeled = self.applyLabels(df_raw)

            # Only flip Benign->Malicious for *specific* hostile tactics
            mask = (
                (df_labeled["Label"] == "Benign") &
                df_labeled["Tactic"].isin(["Defense Evasion", "Persistence"])
            )
            if mask.any():
                print(f"Forcing {mask.sum()} tagged rows from Benign -> Malicious")
                df_labeled.loc[mask, "Label"] = "Malicious"

            # ------------------------------------------------------------
            # 3)  SAVE / UPDATE RAW-PARSED CSV
            # ------------------------------------------------------------
            if self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                if source_is_hive:
                    # New hive data ⇒ append so corpus grows
                    self.appendToExistingCsv(df_labeled, self.sRawParsedCsvPath)
                    print(f"Appended raw-labeled data to: {self.sRawParsedCsvPath}")
                else:
                    # Just refreshing labels ⇒ overwrite
                    df_labeled.to_csv(self.sRawParsedCsvPath, index=False)
                    print(f"Over-wrote raw parsed CSV: {self.sRawParsedCsvPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_raw = os.path.join(self.sModelOutputDir, f"raw_parsed_{ts}.csv")
                df_labeled.to_csv(new_raw, index=False)
                self.sRawParsedCsvPath = new_raw
                print(f"Created new raw-parsed CSV: {new_raw}")
                self.rawParsedCsvInput.delete(0, "end")
                self.rawParsedCsvInput.insert(0, new_raw)

            # ------------------------------------------------------------
            # 4)  PREPROCESS -> TRAINING DATASET
            # ------------------------------------------------------------
            print("Preprocessing for training dataset...")
            df_preproc = self.preprocessData(df_labeled)

            # Keep only model-input columns
            df_preproc = df_preproc[self.selectTrainingColumns(df_preproc)]

            # Fill blanks / NaNs
            df_preproc.replace("", "0", inplace=True)
            df_preproc.fillna("0", inplace=True)

            # Save / append training dataset
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                if source_is_hive:
                    self.appendToExistingCsv(df_preproc, self.sTrainingDatasetPath)
                    print(f"Appended to training dataset: {self.sTrainingDatasetPath}")
                else:
                    df_preproc.to_csv(self.sTrainingDatasetPath, index=False)
                    print(f"Over-wrote training dataset: {self.sTrainingDatasetPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_train = os.path.join(self.sModelOutputDir,
                                         f"training_dataset_{ts}.csv")
                df_preproc.to_csv(new_train, index=False)
                self.sTrainingDatasetPath = new_train
                print(f"Created new training dataset: {new_train}")
                messagebox.showinfo("Dataset Created",
                                    f"New training dataset:\n{new_train}")

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
    def applyLabels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preserve any existing Label / Tactic columns.
        Add / override labels using user-supplied TXT lists.
        Set TagHit when a tagged entry matches.
        Upgrade Benign->Malicious only if TagHit is True and
        the resulting Tactic is one you consider hostile.
        """
        if "Key" not in df.columns:
            raise KeyError("No 'Key' column found in dataframe.")

        # Ensure required columns exist
        df["Label"]  = df.get("Label",  "Benign").fillna("Benign").astype(str)
        df["Tactic"] = df.get("Tactic", "None").fillna("None").astype(str)
        df["TagHit"] = False

        # Read artefact lists
        mal_list = self.read_txt(self.sMaliciousKeysPath)               # may be []
        tag_list = self.read_txt(self.sTaggedKeysPath, "Persistence")   # default tactic

        # -------- 1)  Direct malicious list -> Label = Malicious --------
        if mal_list:
            for idx, row in df.iterrows():
                if row["Label"].strip().lower() == "malicious":
                    continue
                rk, rn, rv, rt = (
                    row["Key"].lower(),
                    str(row.get("Name",  "0")).lower(),
                    str(row.get("Value","0")).lower(),
                    str(row.get("Type",  "0")).lower()
                )
                if any(self.strict_match(ent, rk, rn, rv, rt) for ent in mal_list):
                    df.at[idx, "Label"] = "Malicious"

        # -------- 2)  Tag list -> set Tactic & TagHit -------------------
        if tag_list:
            for idx, row in df.iterrows():
                if row["Tactic"].strip().lower() != "none":
                    continue
                rk, rn, rv, rt = (
                    row["Key"].lower(),
                    str(row.get("Name",  "0")).lower(),
                    str(row.get("Value","0")).lower(),
                    str(row.get("Type",  "0")).lower()
                )
                for ent in tag_list:
                    if self.strict_match(ent, rk, rn, rv, rt):
                        df.at[idx, "Tactic"] = ent["Tactic"]
                        df.at[idx, "TagHit"] = True
                        break

        # -------- 3)  Final Benign->Malicious promotion -----------------
        flip_mask = (
            (df["Label"].str.lower() == "benign") &
            df["TagHit"] &
            df["Tactic"].isin(["Defense Evasion", "Persistence"])
        )
        if flip_mask.any():
            print(f"[INFO] Upgrading {flip_mask.sum()} benign rows due to tag hits.")
            df.loc[flip_mask, "Label"] = "Malicious"

        return df
    
    ###########################################################################
    #                         PREPROCESS (NO RFE)
    ###########################################################################
    def preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates engineered columns, one-hot encodes them,
        then re-indexes so every expected dummy column exists
        all-zero where category absent).
        Scales numeric fields.
        """
        if df.empty:
            return pd.DataFrame()

        xdf = df.copy()
        if "TagHit" not in xdf.columns:
            xdf["TagHit"] = False

        # == engineered cols ==========================================
        xdf["Path Category"]     = xdf["Key"].apply(self.categorizePath)
        xdf["Type Group"]        = xdf["Type"].apply(self.mapType)
        xdf["Key Name Category"] = xdf["Name"].apply(self.categorizeKeyName)
        xdf["Value Processed"]   = xdf["Value"].apply(self.preprocessValue)

        # == expected categories (column universe) ===================
        PATH_CATS  = ["Startup Path", "Service Path", "Network Path", "Other Path"]
        TYPE_GRP   = ["String", "Numeric", "Binary", "Others"]
        KEYNAME_C  = [
            "Run Keys", "Service Keys", "Security and Configuration Keys",
            "Internet and Network Keys", "File Execution Keys", "Other Keys"
        ]

        def fixed_dummies(series, prefix, full):
            d = pd.get_dummies(series, prefix=prefix)
            need_cols = [f"{prefix}_{c}" for c in full]
            return d.reindex(columns=need_cols, fill_value=0)

        xdf = pd.concat([
            xdf,
            fixed_dummies(xdf["Path Category"], "PathCategory", PATH_CATS),
            fixed_dummies(xdf["Type Group"],   "TypeGroup",    TYPE_GRP),
            fixed_dummies(xdf["Key Name Category"], "KeyNameCategory", KEYNAME_C)
        ], axis=1)

        # == scale numeric fields =====================================
        for col in ["Depth", "Value Count", "Value Processed"]:
            if col in xdf.columns:
                xdf[[col]] = MinMaxScaler().fit_transform(xdf[[col]])
        for col in ["Key Size", "Subkey Count"]:
            if col in xdf.columns:
                xdf[[col]] = RobustScaler().fit_transform(xdf[[col]])

        return xdf

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
        If all three model paths and a CSV-to-classify are supplied,
        skip training and go straight to classification.
        Otherwise, run the full train-then-classify pipeline.
        """
        try:
            # -- 0) classify-only short-circuit ---------------------------
            models_ready = all([
                self.sLabelModelPath       and os.path.exists(self.sLabelModelPath),
                self.sTacticModelPath      and os.path.exists(self.sTacticModelPath),
                self.sPersistenceModelPath and os.path.exists(self.sPersistenceModelPath)
            ])
            classify_ready = self.sClassifyCsvPath and os.path.exists(self.sClassifyCsvPath)

            if models_ready and classify_ready:
                print("[ML] Models + classify CSV detected – skipping training.")

                label_model = joblib.load(self.sLabelModelPath)
                if hasattr(label_model, "feature_names_in_"):
                    self.selected_features = list(label_model.feature_names_in_)
                else:
                    feat_path = os.path.join(self.sModelOutputDir, "selected_features.txt")
                    if not os.path.exists(feat_path):
                        raise FileNotFoundError(
                            "selected_features.txt not found – run one full training cycle."
                        )
                    with open(feat_path, encoding="utf-8") as fh:
                        self.selected_features = [l.strip() for l in fh if l.strip()]

                self.classifyCsv(self.sClassifyCsvPath)
                messagebox.showinfo("ML Process Complete", "Classification finished!")
                return

            # ------------------------------------------------------------
            # 1)  Normal training + classification path
            # ------------------------------------------------------------
            def merge_duplicate_columns(df):
                if not df.columns.duplicated().any():
                    return df
                new = pd.DataFrame()
                for name in df.columns.unique():
                    cols = df.loc[:, df.columns == name]
                    new[name] = cols.bfill(axis=1).ffill(axis=1).iloc[:, 0]
                return new

            def remove_non_training_columns(df):
                needed = {
                    "Key", "Label", "Tactic",
                    "Depth", "Key Size", "Subkey Count",
                    "Value Count", "Value Processed",
                    *[f"PathCategory_{c}" for c in
                      ["Startup Path", "Service Path", "Network Path", "Other Path"]],
                    *[f"TypeGroup_{g}" for g in
                      ["String", "Numeric", "Binary", "Others"]],
                    *[f"KeyNameCategory_{k}" for k in [
                        "Run Keys", "Service Keys", "Security and Configuration Keys",
                        "Internet and Network Keys", "File Execution Keys", "Other Keys"
                    ]]
                }
                drop = [c for c in df.columns if c not in needed]
                return df.drop(columns=drop, errors="ignore")

            # ---------- locate or build a training dataset --------------
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                print("[ML] Using provided training dataset.")
                df = self.safe_read_csv(self.sTrainingDatasetPath)

            elif self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                print("[ML] No training dataset. Preprocessing raw-parsed CSV …")
                raw_df = self.safe_read_csv(self.sRawParsedCsvPath)
                raw_df = self.clean_dataframe(raw_df, drop_all_zero_rows=True,
                                              preserve_labels=True)
                df = self.preprocessData(raw_df)
                df = df[self.selectTrainingColumns(df)]

            else:
                raise FileNotFoundError("No training dataset or raw parsed CSV found.")

            # ---------- structural clean-ups ----------------------------
            df = merge_duplicate_columns(df)
            df = remove_non_training_columns(df)
            df = self.clean_dataframe(df, drop_all_zero_rows=True,
                                      preserve_labels=True)

            # ---------- fallback: what to classify after training -------
            if not self.sClassifyCsvPath or not os.path.exists(self.sClassifyCsvPath):
                print("[ML] No classify CSV provided -> will classify training set.")
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            # ---------- train, evaluate, save models --------------------
            self.trainAndEvaluateModels(df)

            # ---------- classify the chosen CSV -------------------------
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete",
                                "Training + classification finished!")

        except Exception as e:
            messagebox.showerror("Error in ML process", str(e))

    # ----------------------------------------------------------------------
    # 2.  Grid-search helpers
    # ----------------------------------------------------------------------
    def label_grid_search_rf(self, Xp, yp):
        brf = BalancedRandomForestClassifier(
            sampling_strategy='all', replacement=True, bootstrap=False, random_state=42
        )
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        gs = GridSearchCV(
            brf, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1, verbose=1
        )
        gs.fit(Xp, yp)
        return gs.best_estimator_

    def tactic_grid_search_rf(self, Xp, yp):
        brf = BalancedRandomForestClassifier(
            sampling_strategy='all', replacement=True, bootstrap=False, random_state=42
        )
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 25],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        gs = GridSearchCV(
            brf, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1, verbose=1
        )
        gs.fit(Xp, yp)
        return gs.best_estimator_


    # ----------------------------------------------------------------------
    # 3.  trainAndEvaluateModels
    # ----------------------------------------------------------------------
    def trainAndEvaluateModels(self, df):
        """
        Train three Balanced-Random-Forest models, evaluate on an 80/20 split,
        save the models and the list of RFE-selected features.
        """
        if df.empty:
            raise ValueError("Training dataset is empty")

        # -- numeric conversion ------------------------------------------
        for c in ['Depth', 'Key Size', 'Subkey Count',
                  'Value Count', 'Value Processed']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # -- feature / target split --------------------------------------
        non_feat = ['Key', 'Name', 'Value', 'Label', 'Tactic',
                    'Type', 'Type Group', 'Key Name Category', 'Path Category']
        X_all = df.drop(columns=[c for c in non_feat if c in df], errors='ignore')
        X_all = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)

        y_label   = (df['Label']  == 'Malicious').astype(int)
        y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
        y_persist = (df['Tactic'] == 'Persistence').astype(int)

        # balance single-class targets
        for y in (y_label, y_defense, y_persist):
            if y.nunique() < 2:
                idx = np.random.choice(y.index, size=int(len(y)*0.3), replace=False)
                y.iloc[idx] = 1

        # -- RFE with replacement=True to silence warning ----------------
        rfe_base = BalancedRandomForestClassifier(
            sampling_strategy='all', replacement=True,  #  ← fixed
            bootstrap=False, n_estimators=100, random_state=42
        )
        rfe = RFE(rfe_base, n_features_to_select=10)
        rfe.fit(X_all, y_label)
        self.selected_features = X_all.columns[rfe.support_]
        X_sel = X_all[self.selected_features]

        # -- train models ------------------------------------------------
        label_model   = self.label_grid_search_rf(X_sel, y_label)
        defense_model = self.tactic_grid_search_rf(X_sel, y_defense)
        persist_model = self.tactic_grid_search_rf(X_sel, y_persist)

        # -- evaluate on 20 % hold-out ----------------------------------
        def evaluate(m, X, y, tag):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=42
            )
            m.fit(X_tr, y_tr)
            y_pred   = m.predict(X_te)
            y_scores = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else None
            return {
                f"{tag} Accuracy":  accuracy_score(y_te, y_pred),
                f"{tag} Precision": precision_score(y_te, y_pred, zero_division=0),
                f"{tag} Recall":    recall_score(y_te, y_pred, zero_division=0),
                f"{tag} F1":        f1_score(y_te, y_pred,  zero_division=0),
                f"{tag} AUC":       (roc_auc_score(y_te, y_scores)
                                     if y_scores is not None and y_te.nunique()>1 else 0.0)
            }

        label_metrics   = evaluate(label_model,   X_sel, y_label,   "Label Model")
        defense_metrics = evaluate(defense_model, X_sel, y_defense, "Defense Model")
        persist_metrics = evaluate(persist_model, X_sel, y_persist, "Persistence Model")

        # -- save models -------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sLabelModelPath       = os.path.join(self.sModelOutputDir, f"label_model_{ts}.joblib")
        self.sTacticModelPath      = os.path.join(self.sModelOutputDir, f"defense_model_{ts}.joblib")
        self.sPersistenceModelPath = os.path.join(self.sModelOutputDir, f"persistence_model_{ts}.joblib")
        joblib.dump(label_model,   self.sLabelModelPath)
        joblib.dump(defense_model, self.sTacticModelPath)
        joblib.dump(persist_model, self.sPersistenceModelPath)

        # -- persist feature list ---------------------------------------
        feat_path = os.path.join(self.sModelOutputDir, "selected_features.txt")
        with open(feat_path, "w", encoding="utf-8") as fh:
            for f in self.selected_features:
                fh.write(f + "\n")
        print(f"[INFO] Selected-feature list saved -> {feat_path}")

        # -- update GUI tables ------------------------------------------
        combined = {label_metrics, defense_metrics, persist_metrics}
        self.updateMetricsDisplay({k: f"{v:.4f}" for k, v in combined.items()})
        self.updateFeatureDisplay(label_model.feature_importances_, self.selected_features)
            
    ###########################################################################
    #                TRAIN AND EVALUATE MODELS
    ###########################################################################
    def label_grid_search_rf(self, Xp, yp):
        """
        Return a BalancedRandomForestClassifier tuned for the Label task.
        """
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        brf = BalancedRandomForestClassifier(sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
        gs = GridSearchCV(
            brf,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        gs.fit(Xp, yp)
        return gs.best_estimator_

    def tactic_grid_search_rf(self, Xp, yp):
        """
        Return a BalancedRandomForestClassifier tuned for the Tactic tasks.
        """
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 25],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        brf = BalancedRandomForestClassifier(sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
        gs = GridSearchCV(
            brf,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        gs.fit(Xp, yp)
        return gs.best_estimator_

    def trainAndEvaluateModels(self, df):
        """
        Train three Balanced-RF models, evaluate, display metrics,
        and save the list of RFE-selected features so it can be
        re-loaded for classification-only runs.
        """
        if df.empty:
            raise ValueError("Training dataset is empty")

        # ---------- preprocessing identical to your original ----------
        for c in ['Depth', 'Key Size', 'Subkey Count',
                  'Value Count', 'Value Processed']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        non_feat = ['Key', 'Name', 'Value', 'Label', 'Tactic',
                    'Type', 'Type Group', 'Key Name Category', 'Path Category']
        X_all = df.drop(columns=[c for c in non_feat if c in df], errors='ignore')
        X_all = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)

        y_label   = (df['Label']   == 'Malicious').astype(int)
        y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
        y_persist = (df['Tactic'] == 'Persistence').astype(int)

        for y in (y_label, y_defense, y_persist):
            if y.nunique() < 2:
                idx = np.random.choice(y.index, size=int(len(y)*0.3), replace=False)
                y.iloc[idx] = 1

        # ---------- RFE ------------------------------
        rfe_base = BalancedRandomForestClassifier(
            sampling_strategy='all', bootstrap=False,
            n_estimators=100, random_state=42
        )
        rfe = RFE(rfe_base, n_features_to_select=10)
        rfe.fit(X_all, y_label)
        self.selected_features = X_all.columns[rfe.support_]
        X_sel = X_all[self.selected_features]

        # ---------- train three models -------------------------------
        label_model   = self.label_grid_search_rf(X_sel, y_label)
        defense_model = self.tactic_grid_search_rf(X_sel, y_defense)
        persist_model = self.tactic_grid_search_rf(X_sel, y_persist)

        # ---------- 20 % hold-out for testing unseen data -------------------------
        def evaluate_model(model, X, y, tag):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=42
            )
            model.fit(X_tr, y_tr)
            y_pred   = model.predict(X_te)
            y_scores = (model.predict_proba(X_te)[:, 1]
                        if hasattr(model, "predict_proba") else None)

            return {
                f"{tag} Accuracy":  accuracy_score(y_te, y_pred),
                f"{tag} Precision": precision_score(y_te, y_pred, zero_division=0),
                f"{tag} Recall":    recall_score(y_te, y_pred, zero_division=0),
                f"{tag} F1":        f1_score(y_te, y_pred,  zero_division=0),
                f"{tag} AUC":       (roc_auc_score(y_te, y_scores)
                                     if y_scores is not None and y_te.nunique() > 1 else 0.0)
            }

        try:
            label_metrics   = evaluate_model(label_model,   X_sel, y_label,   "Label Model")
            defense_metrics = evaluate_model(defense_model, X_sel, y_defense, "Defense Model")
            persist_metrics = evaluate_model(persist_model, X_sel, y_persist, "Persistence Model")
        except Exception as eval_err:
            print(f"[WARN] Metric evaluation failed: {eval_err}")
            label_metrics = defense_metrics = persist_metrics = {}

        # ---------- persist models -----------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sLabelModelPath       = os.path.join(self.sModelOutputDir, f"label_model_{ts}.joblib")
        self.sTacticModelPath      = os.path.join(self.sModelOutputDir, f"defense_model_{ts}.joblib")
        self.sPersistenceModelPath = os.path.join(self.sModelOutputDir, f"persistence_model_{ts}.joblib")
        joblib.dump(label_model,   self.sLabelModelPath)
        joblib.dump(defense_model, self.sTacticModelPath)
        joblib.dump(persist_model, self.sPersistenceModelPath)

        # ---------- save feature list --------------------------------
        feat_path = os.path.join(self.sModelOutputDir, "selected_features.txt")
        with open(feat_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.selected_features))
        print(f"[INFO] Selected-feature list saved -> {feat_path}")

        # ---------- update GUI tables --------------------------------
        combined = {**label_metrics, **defense_metrics, **persist_metrics}
        self.updateMetricsDisplay({k: f"{v:.4f}" for k, v in combined.items()})
        self.updateFeatureDisplay(label_model.feature_importances_, self.selected_features)
    
    ###########################################################################
    #                    CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path: str):
        """
        Preprocess the input CSV if needed, apply the three models,
        and save predictions.  If self.selected_features is missing,
        it is inferred from the label model's `feature_names_in_`.
        """
        try:
            df = self.safe_read_csv(csv_path)

            # -------- raw-versus-training detection --------------------
            if {'Key', 'Name', 'Type'}.issubset(df.columns):
                df = self.clean_dataframe(df, drop_all_zero_rows=True,
                                          preserve_labels=True)
                df = self.preprocessData(df)
                df = df[self.selectTrainingColumns(df)]
            else:
                df = self.clean_dataframe(df, drop_all_zero_rows=True,
                                          preserve_labels=True)

            # -------- load models --------------------------------------
            if not all([
                self.sLabelModelPath       and os.path.exists(self.sLabelModelPath),
                self.sTacticModelPath      and os.path.exists(self.sTacticModelPath),
                self.sPersistenceModelPath and os.path.exists(self.sPersistenceModelPath)
            ]):
                raise FileNotFoundError("One or more trained models not found.")

            model_label   = joblib.load(self.sLabelModelPath)
            model_defense = joblib.load(self.sTacticModelPath)
            model_persist = joblib.load(self.sPersistenceModelPath)

            # -------- ensure we have the feature list ------------------
            if self.selected_features is None:
                if hasattr(model_label, "feature_names_in_"):
                    self.selected_features = list(model_label.feature_names_in_)
                    print(f"[ML] Feature list recovered from model.")
                else:
                    raise RuntimeError(
                        "selected_features list is missing and "
                        "model lacks feature_names_in_."
                    )

            # -------- align & predict ---------------------------------
            df_sel = df[self.selected_features].copy()
            for col in df_sel.columns:
                df_sel[col] = pd.to_numeric(df_sel[col], errors='coerce').fillna(0)

            df['Predicted_Label']            = model_label.predict(df_sel)
            df['Predicted_Tactic_Defense']   = model_defense.predict(df_sel)
            df['Predicted_Tactic_Persistence'] = model_persist.predict(df_sel)

            # -------- save ---------------------------------------------
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.sModelOutputDir,
                                    f"classified_output_{ts}.csv")
            df.to_csv(out_path, index=False)
            print(f"[ML] Saved classified output -> {out_path}")
            messagebox.showinfo("Classification Complete",
                                f"Results saved to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Classification error", str(e))

    ###########################################################################
    #                       UPDATING
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, val in metrics.items():
            self.metricsList.insert("", "end", values=(metric, val))
            
    def updateFeatureDisplay(self, importances, features):
        self.featureList.delete(*self.featureList.get_children())
        feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp:
            self.featureList.insert("", "end", values=(feat, f"{imp:.4f}"))

###########################################################################
#                       MAIN
###########################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
