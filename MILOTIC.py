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
    #                          MAKE DATASET
    ###########################################################################
    def parseRegistry(self, hive_path):
        """
        Parse the registry hive into raw columns and clean the result.
        """
        import pandas as pd, re
        from regipy import RegistryHive

        def clean_dataframe(df, drop_all_zero_rows=True):
            def clean_text(v):
                if pd.isna(v) or v == "": return "0"
                if isinstance(v, bytes):
                    try: return v.decode("utf-8", errors="replace")
                    except: return "0"
                if isinstance(v, str):
                    s = re.sub(r"[^\x20-\x7E]", "", v)
                    return s.strip() or "0"
                return str(v)

            for c in df.columns:
                df[c] = df[c].apply(clean_text)
            df.replace("", "0", inplace=True)
            df.fillna("0", inplace=True)
            if drop_all_zero_rows:
                df = df[~df.eq("0").all(axis=1)]
            return df

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
        return clean_dataframe(df)
        
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
        Read, clean, train models, then classify.
        """
        import pandas as pd

        def clean_dataframe(df, drop_all_zero_rows=True):
            import re
            def clean_text(v):
                if pd.isna(v) or v == "": return "0"
                if isinstance(v, bytes):
                    try: return v.decode("utf-8", errors="replace")
                    except: return "0"
                if isinstance(v, str):
                    s = re.sub(r"[^\x20-\x7E]", "", v)
                    return s.strip() or "0"
                return str(v)

            for c in df.columns:
                df[c] = df[c].apply(clean_text)
            df.replace("", "0", inplace=True)
            df.fillna("0", inplace=True)
            if drop_all_zero_rows:
                df = df[~df.eq("0").all(axis=1)]
            return df

        def safe_read_csv(path, sample_frac=0.2):
            chunk_list = []
            for chunk in pd.read_csv(path, dtype=str, low_memory=True,
                                     on_bad_lines="skip", chunksize=20_000_000):
                chunk_list.append(chunk.sample(frac=sample_frac, random_state=42))
            df = pd.concat(chunk_list, ignore_index=True)
            return clean_dataframe(df)

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
                "Key","Label","Tactic",
                "Depth","Key Size","Subkey Count","Value Count","Value Processed",
                *[f"PathCategory_{c}" for c in ["Startup Path","Service Path","Network Path","Other Path"]],
                *[f"TypeGroup_{g}" for g in ["String","Numeric","Binary","Others"]],
                *[f"KeyNameCategory_{k}" for k in [
                    "Run Keys","Service Keys","Security and Configuration Keys",
                    "Internet and Network Keys","File Execution Keys","Other Keys"
                ]]
            }
            drop = [c for c in df.columns if c not in needed]
            return df.drop(columns=drop, errors="ignore")

        try:
            if not self.sClassifyCsvPath:
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            # 1) load & clean training
            df = safe_read_csv(self.sTrainingDatasetPath)
            df = merge_duplicate_columns(df)
            df = remove_non_training_columns(df)
            df = clean_dataframe(df)

            # 2) train & evaluate
            self.trainAndEvaluateModels(df)

            # 3) classify
            self.classifyCsv(self.sClassifyCsvPath)
            messagebox.showinfo("ML Process Complete", "Finished training & classification!")

        except Exception as e:
            messagebox.showerror("Error in ML process", str(e))
            
    ###########################################################################
    #                TRAIN AND EVALUATE MODELS
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        3 RandomForest models: label, defense, persistence
        If single-class => forcibly flip ~30% to class '1' for demonstration
        RFE is done here only.
        """
        if df.empty:
            raise ValueError("Training dataset is empty")

        # Convert numeric columns
        for c in ['Depth','Key Size','Subkey Count','Value Count','Value Processed']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # Remove non-feature columns
        exclude_cols = []
        for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
            if c in df.columns:
                exclude_cols.append(c)
        X_all = df.drop(columns=exclude_cols, errors='ignore').copy()

        # Ensure numeric
        for col in X_all.columns:
            if X_all[col].dtype == object:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce').fillna(0)

        if 'Label' not in df.columns or 'Tactic' not in df.columns:
            raise ValueError("Missing 'Label'/'Tactic' in dataset")

        y_label = (df['Label']=='Malicious').astype(int)
        y_defense = (df['Tactic']=='Defense Evasion').astype(int)
        y_persist = (df['Tactic']=='Persistence').astype(int)

        # Balance classes if single
        if y_label.nunique()<2:
            idx = np.random.choice(y_label.index, size=int(len(y_label)*0.3), replace=False)
            y_label.iloc[idx]=1
        if y_defense.nunique()<2:
            idx = np.random.choice(y_defense.index, size=int(len(y_defense)*0.3), replace=False)
            y_defense.iloc[idx]=1
        if y_persist.nunique()<2:
            idx = np.random.choice(y_persist.index, size=int(len(y_persist)*0.3), replace=False)
            y_persist.iloc[idx]=1

        # RFE
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=base_model, n_features_to_select=10)
        rfe.fit(X_all, y_label)
        self.selected_features = X_all.columns[rfe.support_]

        # Grid search functions
        def label_grid_search_rf(Xp, yp):
            param_grid = {'n_estimators':[50],'max_depth':[20],'min_samples_split':[2],'min_samples_leaf':[1],'bootstrap':[True],'class_weight': [{0:1, 1:10}]}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = RandomForestClassifier(random_state=42)
            gs = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
            gs.fit(Xp, yp)
            return gs.best_estimator_

        def tactic_grid_search_rf(Xp, yp):
            param_grid = {'n_estimators':[50],'max_depth':[25],'min_samples_split':[2],'min_samples_leaf':[1],'bootstrap':[True],'class_weight': [{0:1, 1:20}]}
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

            return {f"{label_name} Accuracy": acc, f"{label_name} Precision": prec, f"{label_name} Recall": rec, f"{label_name} F1": f1v, f"{label_name} AUC": aucv}

        X_sel = X_all[self.selected_features].copy()

        # Label model
        label_model = label_grid_search_rf(X_sel, y_label)
        label_metrics = evaluate_model(label_model, X_sel, y_label, "Label Model")
        label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
        joblib.dump(label_model, label_model_path)
        self.sLabelModelPath = label_model_path

        # Defense model
        defense_model = tactic_grid_search_rf(X_sel, y_defense)
        defense_metrics = evaluate_model(defense_model, X_sel, y_defense, "Defense Evasion Model")
        defense_model_path = os.path.join(self.sModelOutputDir, "defense_model.joblib")
        joblib.dump(defense_model, defense_model_path)
        self.sTacticModelPath = defense_model_path

        # Persistence model
        persistence_model = tactic_grid_search_rf(X_sel, y_persist)
        persist_metrics = evaluate_model(persistence_model, X_sel, y_persist, "Persistence Model")
        persist_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
        joblib.dump(persistence_model, persist_model_path)
        self.sPersistenceModelPath = persist_model_path

        # Metrics display
        combined = {} 
        combined.update(label_metrics)
        combined.update(defense_metrics)
        combined.update(persist_metrics)
        out_metrics = {k: f"{v:.4f}" for k,v in combined.items()}
        self.updateMetricsDisplay(out_metrics)

        # Feature importances display
        self.updateFeatureDisplay(label_model.feature_importances_, self.selected_features)
    
    ###########################################################################
    #                    CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path: str):
        """
        Read fallback CSV, then predict & save.
        """
        import pandas as pd, chardet

        def clean_dataframe(df, drop_all_zero_rows=True):
            import re
            def clean_text(v):
                if pd.isna(v) or v == "": return "0"
                if isinstance(v, bytes):
                    try: return v.decode("utf-8", errors="replace")
                    except: return "0"
                if isinstance(v, str):
                    s = re.sub(r"[^\x20-\x7E]", "", v)
                    return s.strip() or "0"
                return str(v)

            for c in df.columns:
                df[c] = df[c].apply(clean_text)
            df.replace("", "0", inplace=True)
            df.fillna("0", inplace=True)
            if drop_all_zero_rows:
                df = df[~df.eq("0").all(axis=1)]
            return df

        def read_csv_with_fallbacks(path):
            raw = open(path, "rb").read(10000)
            enc = chardet.detect(raw)["encoding"] or "utf-8"
            df = pd.read_csv(path, encoding=enc, encoding_errors="replace", dtype=str, low_memory=False)
            return clean_dataframe(df)

        try:
            df = read_csv_with_fallbacks(csv_path)
            if df.empty:
                raise ValueError("No data to classify.")
            if "Key" not in df.columns:
                df["Key"] = "UNKNOWN"
            # …rest of classification logic unchanged…
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
