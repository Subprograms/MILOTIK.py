import os
import tkinter as tk
import pandas as pd, chardet
import numpy as np
import joblib
import re
import chardet
import matplotlib.pyplot as plt

from matplotlib import rcParams
from io import BytesIO
from PIL import Image, ImageTk, Image as _PIL_Image
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, messagebox
from regipy import RegistryHive
from datetime import datetime
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.feature_selection import RFE

###########################################################################
#                           MAIN MILOTIC CLASS
###########################################################################
class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("1600x780")
        self.root.resizable(True, True)
        
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
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, sticky="nsew")

        for r in range(0, 13):
            frame.rowconfigure(r, weight=0)
            frame.rowconfigure(10, weight=1)  # metrics list grows
            frame.rowconfigure(11, weight=1)  # feature list grows
            frame.columnconfigure(0, weight=0)
            frame.columnconfigure(1, weight=0)
            frame.columnconfigure(2, weight=0)
            frame.columnconfigure(3, weight=1)  # tree view grows

        entry_opts = {"width": 50}
        label_opts = {"sticky": "e", "padx": 2, "pady": 1}
        entry_pad = {"padx": 2, "pady": 1, "sticky": "w"}
        btn_pad = {"padx": 2, "pady": 1, "sticky": "w"}

        ttk.Label(frame, text="Hive Path:").grid(row=0, column=0, **label_opts)
        self.hivePathInput = ttk.Entry(frame, **entry_opts)
        self.hivePathInput.grid(row=0, column=1, **entry_pad)
        ttk.Button(frame, text="Set Hive Path", command=self.setHivePath).grid(row=0, column=2, **btn_pad)

        ttk.Label(frame, text="Malicious Keys File:").grid(row=1, column=0, **label_opts)
        self.maliciousKeysInput = ttk.Entry(frame, **entry_opts)
        self.maliciousKeysInput.grid(row=1, column=1, **entry_pad)
        ttk.Button(frame, text="Set Malicious Keys", command=self.setMaliciousKeysPath).grid(row=1, column=2, **btn_pad)

        ttk.Label(frame, text="Tagged Keys File:").grid(row=2, column=0, **label_opts)
        self.taggedKeysInput = ttk.Entry(frame, **entry_opts)
        self.taggedKeysInput.grid(row=2, column=1, **entry_pad)
        ttk.Button(frame, text="Set Tagged Keys", command=self.setTaggedKeysPath).grid(row=2, column=2, **btn_pad)

        ttk.Label(frame, text="Training Dataset (Optional):").grid(row=3, column=0, **label_opts)
        self.trainingDatasetInput = ttk.Entry(frame, **entry_opts)
        self.trainingDatasetInput.grid(row=3, column=1, **entry_pad)
        ttk.Button(frame, text="Set Training Dataset", command=self.setTrainingDatasetPath).grid(row=3, column=2, **btn_pad)

        ttk.Label(frame, text="Raw Parsed CSV (Optional):").grid(row=4, column=0, **label_opts)
        self.rawParsedCsvInput = ttk.Entry(frame, **entry_opts)
        self.rawParsedCsvInput.grid(row=4, column=1, **entry_pad)
        ttk.Button(frame, text="Set Raw Parsed CSV", command=self.setRawParsedCsvPath).grid(row=4, column=2, **btn_pad)

        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=5, column=0, **label_opts)
        self.classifyCsvInput = ttk.Entry(frame, **entry_opts)
        self.classifyCsvInput.grid(row=5, column=1, **entry_pad)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=5, column=2, **btn_pad)

        ttk.Label(frame, text="Label Model (Optional):").grid(row=6, column=0, **label_opts)
        self.labelModelInput = ttk.Entry(frame, **entry_opts)
        self.labelModelInput.grid(row=6, column=1, **entry_pad)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=6, column=2, **btn_pad)

        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=7, column=0, **label_opts)
        self.tacticModelInput = ttk.Entry(frame, **entry_opts)
        self.tacticModelInput.grid(row=7, column=1, **entry_pad)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=7, column=2, **btn_pad)

        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=8, column=0, **label_opts)
        self.persistenceModelInput = ttk.Entry(frame, **entry_opts)
        self.persistenceModelInput.grid(row=8, column=1, **entry_pad)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=8, column=2, **btn_pad)

        # Pipeline Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10, sticky="w")
        
        ttk.Button(btn_frame, text="Make Dataset", command=self.makeDataset).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Start ML Process", command=self.executeMLProcess).pack(side="left", padx=5)
        
        self.rfeInputs = {}
        for tag in ["Label", "Defense", "Persistence"]:
            ttk.Label(btn_frame, text=f"RFE % for {tag}:").pack(side="left", padx=(15, 2))
            entry = ttk.Entry(btn_frame, width=5, foreground="gray")
            entry.insert(0, "50")
            entry.pack(side="left")

            def clearText(e=entry):
                return lambda event: (e.delete(0, "end"), e.config(foreground="black"))

            entry.bind("<FocusIn>", clearText())
            self.rfeInputs[tag] = entry

        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=10)
        self.metricsList = ttk.Treeview(metrics_frame, columns=("Metric", "Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value",  text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value",  width=500, anchor="w")
        self.metricsList.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metricsList.yview).pack(side="right", fill="y")

        feature_frame = ttk.Frame(frame)
        feature_frame.grid(row=11, column=0, columnspan=3, sticky="nsew")

        self.featureTabs = ttk.Notebook(feature_frame)
        self.featureTabs.pack(fill="both", expand=True)

        self.featureTabsTabs = {}
        for tag in ("Label", "Defense", "Persistence"):
            tab = ttk.Frame(self.featureTabs)
            self.featureTabs.add(tab, text=tag)
            tree = ttk.Treeview(tab, columns=("Feature", "Importance"), show="headings")
            tree.heading("Feature", text="Feature")
            tree.heading("Importance", text="Importance")
            tree.column("Feature", width=200, anchor="w")
            tree.column("Importance", width=120, anchor="w")
            tree.pack(fill="both", expand=True)
            self.featureTabsTabs[tag] = tree

        self.treeNotebook = ttk.Notebook(frame)
        self.treeNotebook.grid(row=0, column=3, rowspan=12, sticky="nsew", padx=10, pady=5)

        self.img_labels = {}
        for tag in ("Label", "Defense", "Persistence"):
            tab = ttk.Frame(self.treeNotebook)
            self.treeNotebook.add(tab, text=tag)
            lbl = tk.Label(tab, bg="white")
            lbl.pack(fill="both", expand=True)
            self.img_labels[tag] = lbl

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
        - Exact path match by default  
        - Allow trailing '*' in artefact path to act as “starts-with” wildcard  
        - Optional Name / Value / Type must also match if present
        """
        kp = ent["Key"]
        if kp.endswith("*"):
            # wildcard -> prefix match
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
    #    return self.cleanDataframe(df_raw, drop_all_zero_rows=True, preserve_labels=True)
    
    def zoomCanvas(self, tag):
        """
        Replace the (tab, Label) pair created in setupUI with a scroll-zoom Canvas.
        Call this once per tag when you first draw that tree.
        """
        if tag in getattr(self, "_tree_canvases", {}):
            return

        lbl      = self.img_labels[tag]
        parent   = lbl.master # the Notebook tab
        lbl.destroy()

        c       = tk.Canvas(parent, bg="white")
        vbar    = ttk.Scrollbar(parent, orient="vertical",
                                command=c.yview)
        hbar    = ttk.Scrollbar(parent, orient="horizontal",
                                command=c.xview)
        c.configure(yscrollcommand=vbar.set,
                    xscrollcommand=hbar.set)

        c.grid (row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        c.bind("<MouseWheel>", lambda e, t=tag: self.onZoom(e, t))
        c.bind("<ButtonPress-1>",lambda e, canv=c: canv.scan_mark(e.x, e.y))
        c.bind("<B1-Motion>",    lambda e, canv=c: canv.scan_dragto(e.x, e.y, 1))

        self._tree_canvases = getattr(self, "_tree_canvases", {})
        self._tree_canvases[tag] = {
            "canvas"   : c,
            "pil_orig" : None, # full-res Pillow image
            "tk_img"   : None,
            "scale"    : 1.0,
        }

    def onZoom(self, event, tag):
        """
        Mouse-wheel zoom – loss-less because we always resample from the
        stored full-resolution bitmap.
        """
        rec         = self._tree_canvases[tag]
        pil_full    = rec["pil_orig"]

        # wheel direction for scrolling
        delta       = 1.1 if event.delta > 0 else (1/1.1)
        new_scale   = rec["scale"] * delta
        new_scale   = min(max(new_scale, 0.1), 5.0)
        rec["scale"] = new_scale

        new_w, new_h = int(pil_full.width  * new_scale), \
                       int(pil_full.height * new_scale)

        from PIL import Image
        preview = pil_full.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling") else Image.LANCZOS
        )
        from PIL import ImageTk
        tk_img  = ImageTk.PhotoImage(preview)
        rec["tk_img"] = tk_img

        canv = rec["canvas"]
        canv.delete("all")
        canv.create_image(0, 0, anchor="nw", image=tk_img)
        canv.config(scrollregion=(0, 0, new_w, new_h))

    ###########################################################################
    #                          MAKE DATASET
    ###########################################################################
    def cleanDataframe(self, df: pd.DataFrame, drop_all_zero_rows: bool = True, preserve_labels: bool = True) -> pd.DataFrame:
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

        # basic cleaning and filling
        for col in df.columns:
            df[col] = df[col].apply(clean_text)
        df.replace("", "0", inplace=True)
        df.fillna("0", inplace=True)

        # if needed, fill missing Label and Tactic columns with defaults
        if preserve_labels:
            if "Label" not in df.columns:
                df["Label"] = "Unknown"
            if "Tactic" not in df.columns:
                df["Tactic"] = "Unknown"

        # optional row prune
        if drop_all_zero_rows:
            numeric_ok = df.select_dtypes(include=[np.number]).sum(axis=1) > 0
            string_ok  = df.select_dtypes(include=[object]).ne("0").any(axis=1)
            keep_mask  = numeric_ok | string_ok
            if preserve_labels:
                keep_mask |= df["Label"].astype(str).str.lower().eq("malicious")
                keep_mask |= df["Tactic"].astype(str).str.lower().ne("none")
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
        return self.cleanDataframe(df, drop_all_zero_rows=True, preserve_labels=True)
        
    def makeDataset(self):
        """
        1) Load either a hive (parseRegistry) or a raw CSV (safe_read_csv).
        2) Clean + label.
        3) Save a new raw-labeled CSV if loading from raw CSV.
        4) Preprocess into a training dataset.
        5) Save training CSV and update the entry box.
        """
        try:
            # 1) LOAD SOURCE DATA
            source_is_hive = bool(self.sHivePath and os.path.exists(self.sHivePath))

            if source_is_hive:
                df_raw = self.parseRegistry(self.sHivePath)
            elif self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                df_raw = self.safe_read_csv(self.sRawParsedCsvPath)
            else:
                raise FileNotFoundError("No valid Hive Path or Raw Parsed CSV provided.")

            # 2) CLEAN & LABEL
            df_clean   = self.cleanDataframe(df_raw, drop_all_zero_rows=True, preserve_labels=True)
            df_labeled = self.applyLabels(df_clean)

            # 3) SAVE / UPDATE RAW-LABELED CSV
            if not source_is_hive:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_raw = os.path.join(self.sModelOutputDir, f"raw_labeled_{ts}.csv")
                df_labeled.to_csv(new_raw, index=False)
                self.sRawParsedCsvPath = new_raw
                self.rawParsedCsvInput.delete(0, tk.END)
                self.rawParsedCsvInput.insert(0, new_raw)
                print(f"Created new raw-labeled CSV: {new_raw}")

            # 4) PREPROCESS -> TRAINING DATASET
            if any(col in df_labeled.columns for col in ("Name", "Value", "Type")):
                df_preproc = self.preprocessData(df_labeled)
                df_preproc = df_preproc[self.selectTrainingColumns(df_preproc)]
            else:
                print("[makeDataset] Input appears preprocessed — skipping `preprocessData()`.")
                df_preproc = df_labeled[self.selectTrainingColumns(df_labeled)]

            # 5) SAVE TRAINING DATASET AND UPDATE UI FIELD
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_train = os.path.join(self.sModelOutputDir, f"training_dataset_{ts}.csv")
            df_preproc.to_csv(new_train, index=False)
            self.sTrainingDatasetPath = new_train
            self.trainingDatasetInput.delete(0, tk.END)
            self.trainingDatasetInput.insert(0, new_train)
            messagebox.showinfo("Dataset Created", f"New training dataset:\n{new_train}")
            print(f"Created new training dataset: {new_train}")

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
          - dummy columns that start with PathCategory_, TypeGroup_, KeyNameCategory_, TypeCategory_
          - and ensure depth-based PathCategory_*_Depth* columns come right after general PathCategory_ ones
        """
        keep_cols = []

        # Column identification
        for c in df.columns:
            if c == 'Key':
                keep_cols.append(c)
            elif c in ('Label','Tactic'):
                keep_cols.append(c)
            elif c in ('Depth','Key Size','Subkey Count','Value Count','Value Processed'):
                keep_cols.append(c)

        # Separate general path categories from the specific ones for ordering
        path_cat_cols      = [c for c in df.columns if c.startswith('PathCategory_') and '_Depth' not in c]
        path_specific_cols = [c for c in df.columns if c.startswith('PathCategory_') and '_Depth' in c]
        keep_cols += path_cat_cols + path_specific_cols

        # Include remaining feature–engineered dummies, now with TypeCategory_
        for c in df.columns:
            if c.startswith(('TypeGroup_', 'KeyNameCategory_', 'TypeCategory_')):
                keep_cols.append(c)

        # Drop anything that slipped in
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

        # 1) Direct malicious list -> Label = Malicious
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

        # 2) Tag list -> set Tactic & TagHit
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

        # 3) Final Benign->Malicious promotion
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
    #                         PREPROCESS (NO RFE YET)
    ###########################################################################
    def preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates engineered columns, one-hot encodes them with uniform prefixes,
        then re-indexes so every expected dummy column exists (all-zero where absent),
        and finally scales numeric fields. Rare dummy columns are grouped or dropped.
        """
        for col in ("Name", "Value", "Type"):
            if col not in df.columns:
                df[col] = "0"

        if df.empty:
            return pd.DataFrame()

        xdf = df.copy()
        xdf.drop(columns=["TagHit"], inplace=True, errors="ignore")

        # 1) original engineered columns
        xdf["Path Category"]     = xdf["Key"].apply(self.categorizePath)
        xdf["Key Name Category"] = xdf["Name"].apply(self.categorizeKeyName)
        xdf["Value Processed"]   = xdf["Value"].apply(self.preprocessValue)

        # 2) define the full lists for stable ordering
        PATH_CATS  = ["Startup Path", "Service Path", "Network Path", "Other Path"]
        KEYNAME_C  = [
            "Run Keys", "Service Keys", "Security and Configuration Keys",
            "Internet and Network Keys", "File Execution Keys", "Other Keys"
        ]
        KNOWN_TYPES = [
            "REG_SZ", "REG_EXPAND_SZ", "REG_MULTI_SZ",
            "REG_DWORD", "REG_QWORD",
            "REG_BINARY",
            "REG_NONE", "REG_LINK"
        ]

        def fixedDummies(series, prefix, full):
            d = pd.get_dummies(series, prefix=prefix)
            need_cols = [f"{prefix}_{c}" for c in full]
            return d.reindex(columns=need_cols, fill_value=0)

        # 3) general category dummies
        df_general = pd.concat([
            fixedDummies(xdf["Path Category"],     "PathCategory",  PATH_CATS),
            fixedDummies(xdf["Type"],              "TypeCategory",  KNOWN_TYPES),
            fixedDummies(xdf["Key Name Category"], "NameCategory",  KEYNAME_C),
        ], axis=1)

        # 4) specific per-path and per-name features
        from collections import Counter

        # Limit high-cardinality PathCategory_*_Depth features
        path_parts_counter = Counter()
        for p in xdf["Key"]:
            parts = [part for part in p.strip("\\").split("\\") if part]
            path_parts_counter.update(parts)

        common_parts = set(k for k, v in path_parts_counter.items() if v >= 5)

        def makeSpecificPathFeatures(p):
            parts = [part for part in p.strip("\\").split("\\") if part]
            out = {}
            for i, part in enumerate(parts):
                key = part if part in common_parts else "Other"
                out[f"PathCategory_{key}_Depth{i}"] = 1
            return pd.Series(out, dtype="float64")

        # Limit high-cardinality NameCategory_* features
        name_counter = Counter(xdf["Name"])
        common_names = set(k for k, v in name_counter.items() if v >= 5)

        def makeSpecificKeynameFeatures(n):
            if not isinstance(n, str):
                return pd.Series(dtype="float64")
            name = n if n in common_names else "Other"
            cleaned = re.sub(r"[^\w]", "_", name.strip())
            return pd.Series({f"NameCategory_{cleaned}": 1}, dtype="float64")

        path_feats = xdf["Key"].apply(makeSpecificPathFeatures).fillna(0)
        name_feats = xdf["Name"].apply(makeSpecificKeynameFeatures).fillna(0)

        # 5) combine all feature sets
        xdf = pd.concat([
            xdf.drop(columns=["Path Category", "Key Name Category"], errors="ignore"),
            df_general,
            path_feats,
            name_feats
        ], axis=1)

        # 6) scale numeric fields
        for col in ["Depth", "Value Count", "Value Processed"]:
            if col in xdf.columns:
                xdf[[col]] = MinMaxScaler().fit_transform(xdf[[col]])
        for col in ["Key Size", "Subkey Count"]:
            if col in xdf.columns:
                xdf[[col]] = RobustScaler().fit_transform(xdf[[col]])

        # 7) drop ultra-sparse dummy columns (appear < 5 times total)
        dummy_thresh = 5
        xdf = xdf.loc[:, (xdf != 0).sum(axis=0) >= dummy_thresh]

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
        Loads training dataset, cleans, performs early top-100 reduction per model,
        writes preprocessed dataset if needed, and runs ML pipeline.
        """
        try:
            if not self.sTrainingDatasetPath:
                messagebox.showerror("Missing Input", "Please select a training dataset first.")
                return

            print("[ML] Using supplied training dataset.")
            df_raw = pd.read_csv(self.sTrainingDatasetPath, dtype=str, engine="python", on_bad_lines="skip")

            print("[DEBUG] Reading CSV...")
            if df_raw.columns.duplicated().any():
                print("[WARNING] Duplicate columns found. Resolving...")
                df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

            print("[DEBUG] CSV Loaded: shape =", df_raw.shape)

            is_preprocessed = all(col in df_raw.columns for col in ("Depth", "Key Size", "Value Processed"))

            if is_preprocessed:
                print("[DEBUG] Dataset appears preprocessed. Skipping redundant processing.")
            else:
                df_raw = self.cleanDataframe(df_raw, drop_all_zero_rows=True, preserve_labels=True)

            # ------------------ EARLY PER-MODEL FEATURE REDUCTION -----------------------
            print("[DEBUG] Starting early tree-based feature reduction (per model)...")

            def get_top_100(df, target_col, target_val):
                X = df.drop(columns=["Key", "Label", "Tactic"], errors="ignore")
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                y = (df.get(target_col) == target_val).astype(int)

                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                top_100_idx = np.argsort(rf.feature_importances_)[-100:]
                return list(X.columns[top_100_idx])

            top_lbl_cols = get_top_100(df_raw, "Label", "Malicious")
            top_def_cols = get_top_100(df_raw, "Tactic", "Defense Evasion")
            top_per_cols = get_top_100(df_raw, "Tactic", "Persistence")

            print(f"[DEBUG] Top Label features: {len(top_lbl_cols)}")
            print(f"[DEBUG] Top Defense features: {len(top_def_cols)}")
            print(f"[DEBUG] Top Persistence features: {len(top_per_cols)}")

            # Save 3 reduced CSVs for RFE and model training
            all_keep = {
                "Label":       sorted(set(top_lbl_cols + ["Key", "Label", "Tactic"])),
                "Defense":     sorted(set(top_def_cols + ["Key", "Label", "Tactic"])),
                "Persistence": sorted(set(top_per_cols + ["Key", "Label", "Tactic"]))
            }

            for tag, cols in all_keep.items():
                out_path = os.path.splitext(self.sTrainingDatasetPath)[0] + f"_{tag.lower()}_reduced.csv"
                df_raw.loc[:, df_raw.columns.intersection(cols)].to_csv(out_path, index=False)
                print(f"[DEBUG] Saved {tag} reduced dataset to {out_path}")

            # Only save preprocessed version if it was originally raw
            if not is_preprocessed:
                self.sPreprocessedCsvPath = os.path.splitext(self.sTrainingDatasetPath)[0] + "_preprocessed.csv"
                df_raw.to_csv(self.sPreprocessedCsvPath, index=False)
                if hasattr(self, 'txtProcessedDatasetPath'):
                    self.txtProcessedDatasetPath.delete(0, "end")
                    self.txtProcessedDatasetPath.insert(0, self.sPreprocessedCsvPath)

            print("[DEBUG] Launching training + evaluation pipeline...")
            self.trainAndEvaluateModels(df_raw)

        except Exception as e:
            print("[ERROR] executeMLProcess():", e)
            messagebox.showerror("Execution Error", str(e))

    ###########################################################################
    #            TRAIN AND EVALUATE MODELS
    ###########################################################################
    @staticmethod
    def optimalThreshold(y_true, y_score):
        """
        Given true binary labels and predicted scores, find the threshold
        on the ROC curve closest to the top-left (0,1) point.
        """
        from sklearn.metrics import roc_curve
        import numpy as np

        fpr, tpr, thr = roc_curve(y_true, y_score)
        dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        return thr[np.argmin(dist)]
    
    def trainAndEvaluateModels(self, df: pd.DataFrame):
        """
        Run RFE on pre-reduced feature set -> train three BRF models -> evaluate &
        push metrics to GUI -> compute/store optimal ROC thresholds -> dump .joblib
        models -> update feature-importance tabs -> draw zoomable trees.
        """
        if df.empty:
            raise ValueError("Training dataset is empty")

        # 1) Coerce known numeric columns
        for col in ["Depth", "Key Size", "Subkey Count", "Value Count", "Value Processed"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 2) Prepare features and labels
        drop_cols = ["Key", "Label", "Tactic"]
        X_all = df.drop(columns=drop_cols, errors="ignore")
        for col in X_all.columns:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
        X_all.fillna(0, inplace=True)

        y_lbl = (df.get("Label") == "Malicious").astype(int)
        y_def = (df.get("Tactic") == "Defense Evasion").astype(int)
        y_per = (df.get("Tactic") == "Persistence").astype(int)

        # 3) Train-test split
        X_tr, X_te, y_lbl_tr, y_lbl_te = train_test_split(X_all, y_lbl, test_size=0.2, stratify=y_lbl, random_state=42)
        y_def_tr, y_def_te = y_def.loc[X_tr.index], y_def.loc[X_te.index]
        y_per_tr, y_per_te = y_per.loc[X_tr.index], y_per.loc[X_te.index]

        # 4) RFE % configuration
        def getN(tag):
            txt = self.rfeInputs[tag].get().strip().rstrip("%")
            pct = float(txt) if txt else 0.0
            return max(1, int(np.ceil((pct / 100) * X_tr.shape[1])))

        n_lbl = getN("Label")
        n_def = getN("Defense")
        n_per = getN("Persistence")

        # 5) Run RFE
        def runRFE(X, y, n):
            clf = BalancedRandomForestClassifier(n_estimators=400, random_state=42)
            selector = RFE(estimator=clf, n_features_to_select=n, verbose=1)
            selector.fit(X, y)
            return X.columns[selector.support_]

        feats_lbl = runRFE(X_tr, y_lbl_tr, n_lbl)
        feats_def = runRFE(X_tr, y_def_tr, n_def)
        feats_per = runRFE(X_tr, y_per_tr, n_per)

        # 6) Final training sets
        X_tr_lbl, X_te_lbl = X_tr[feats_lbl], X_te[feats_lbl]
        X_tr_def, X_te_def = X_tr[feats_def], X_te[feats_def]
        X_tr_per, X_te_per = X_tr[feats_per], X_te[feats_per]

        clf = lambda: BalancedRandomForestClassifier(
            n_estimators=400, random_state=42,
            sampling_strategy="all", replacement=True,
            bootstrap=False
        )

        m_lbl = clf().fit(X_tr_lbl, y_lbl_tr)
        m_def = clf().fit(X_tr_def, y_def_tr)
        m_per = clf().fit(X_tr_per, y_per_tr)

        # 7) Compute optimal thresholds
        def find_thresh(y_true, scores):
            fpr, tpr, thr = roc_curve(y_true, scores)
            dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
            return thr[np.argmin(dist)]

        prob_lbl = m_lbl.predict_proba(X_te_lbl)[:, 1]
        prob_def = m_def.predict_proba(X_te_def)[:, 1]
        prob_per = m_per.predict_proba(X_te_per)[:, 1]

        self.optimal_thresholds = {
            "Label":       find_thresh(y_lbl_te, prob_lbl),
            "Defense":     find_thresh(y_def_te, prob_def),
            "Persistence": find_thresh(y_per_te, prob_per)
        }

        # 8) Save models
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sLabelModelPath       = os.path.join(self.sModelOutputDir, f"label_model_{ts}.joblib")
        self.sTacticModelPath      = os.path.join(self.sModelOutputDir, f"defense_model_{ts}.joblib")
        self.sPersistenceModelPath = os.path.join(self.sModelOutputDir, f"persistence_model_{ts}.joblib")
        joblib.dump(m_lbl, self.sLabelModelPath)
        joblib.dump(m_def, self.sTacticModelPath)
        joblib.dump(m_per, self.sPersistenceModelPath)

        # 9) Push metrics
        all_metrics = {}
        for tag, mdl, X_e, y_e, pr in [
            ("Label",       m_lbl, X_te_lbl, y_lbl_te, prob_lbl),
            ("Defense",     m_def, X_te_def, y_def_te, prob_def),
            ("Persistence", m_per, X_te_per, y_per_te, prob_per),
        ]:
            all_metrics[f"{tag} Accuracy"]  = f"{accuracy_score(y_e, mdl.predict(X_e)):.4f}"
            all_metrics[f"{tag} Precision"] = f"{precision_score(y_e, mdl.predict(X_e), zero_division=0):.4f}"
            all_metrics[f"{tag} Recall"]    = f"{recall_score(y_e, mdl.predict(X_e), zero_division=0):.4f}"
            all_metrics[f"{tag} F1"]        = f"{f1_score(y_e, mdl.predict(X_e), zero_division=0):.4f}"
            all_metrics[f"{tag} AUC"]       = f"{roc_auc_score(y_e, pr):.4f}"
            all_metrics[f"{tag} Threshold"] = f"{self.optimal_thresholds[tag]:.4f}"

        self.updateMetricsDisplay(all_metrics)

        # 10) Feature tab display
        self.updateFeatureDisplay(
            list(feats_lbl), m_lbl.feature_importances_,
            list(feats_def), m_def.feature_importances_,
            list(feats_per), m_per.feature_importances_,
        )

        # 11) Visualize top trees
        self.showForestTree(m_lbl,      "Label",       feats_lbl, X_tr_lbl, y_lbl_tr)
        self.showForestTree(m_def,      "Defense",     feats_def, X_tr_def, y_def_tr)
        self.showForestTree(m_per,      "Persistence", feats_per, X_tr_per, y_per_tr)

    def labelGridSearch(self, Xp, yp):
        """
        Return a BalancedRandomForestClassifier tuned for the Label task.
        """
        param_grid = {
            "n_estimators": [50],
            "max_depth": [None, 25],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "bootstrap":         [False],
            "class_weight":      [{0:1, 1:10}]
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

    def tacticGridSearch(self, Xp, yp):
        """
        Return a BalancedRandomForestClassifier tuned for the Tactic tasks.
        """
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None, 25],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "bootstrap":         [False],
            "class_weight":      [{0:1, 1:10}]
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

    def showForestTree(self, forest, tag: str, feature_names, X_train, y_train):
        """
        Render the best tree from `forest` into the zoomable canvas
        for the given tag.  Always reset to a uniform default zoom
        that fits the canvas width exactly.
        """
        # pick the best single tree (highest trainset score)
        scores = [est.score(X_train, y_train) for est in forest.estimators_]
        best = forest.estimators_[int(np.argmax(scores))]

        # plot it at high resolution
        rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(24, 16), dpi=600)
        tree.plot_tree(
            best,
            feature_names=None,
            class_names=["Benign/Pure", "Positive"],
            filled=True, rounded=True,
            impurity=False, proportion=False,
            max_depth=5,  # cap depth
            ax=ax
        )
        ax.axis("off")
        fig.tight_layout(pad=0.3)

        # grab it into a Pillow image
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_full = Image.open(buf)

        self.zoomCanvas(tag)
        rec = self._tree_canvases[tag]
        canv = rec["canvas"]

        # reset any old scale
        rec["scale"] = 1.0
        rec["pil_orig"] = pil_full

        canv.update_idletasks()

        cw = canv.winfo_width()
        if cw <= 1:
            fit_scale = 1.0
        else:
            fit_scale = cw / pil_full.width

        rec["scale"] = fit_scale
        new_w  = int(pil_full.width  * fit_scale)
        new_h  = int(pil_full.height * fit_scale)

        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        preview = pil_full.resize((new_w, new_h), resample)
        tk_img  = ImageTk.PhotoImage(preview)
        rec["tk_img"] = tk_img

        # draw
        canv.delete("all")
        canv.create_image(0, 0, anchor="nw", image=tk_img)
        canv.config(scrollregion=(0, 0, new_w, new_h))
    
    ###########################################################################
    #                    CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path: str):
        """
        Load a CSV (raw registry dump or preprocessed), apply the stored models
        using the optimal ROC-based thresholds, update metrics, and save output.
        """
        try:
            # 1) Load raw CSV
            df = self.safe_read_csv(csv_path)

            # 2) Preprocess if it's a raw registry dump
            if {"Key", "Name", "Type"}.issubset(df.columns):
                df = self.cleanDataframe(df, drop_all_zero_rows=True, preserve_labels=True)
                df = self.preprocessData(df)
                df = df[self.selectTrainingColumns(df)]
            else:
                # Already a preprocessed CSV
                df = self.cleanDataframe(df, drop_all_zero_rows=True, preserve_labels=True)

            # 3) Load trained models
            model_label   = joblib.load(self.sLabelModelPath)
            model_defense = joblib.load(self.sTacticModelPath)
            model_persist = joblib.load(self.sPersistenceModelPath)

            # 4) Grab the exact feature lists the models were trained on
            feat_lbl = list(model_label.feature_names_in_)
            feat_def = list(model_defense.feature_names_in_)
            feat_per = list(model_persist.feature_names_in_)

            # 5) Reindex DataFrame to those feature sets (fill missing cols with 0)
            X_lbl = df.reindex(columns=feat_lbl, fill_value=0)
            X_def = df.reindex(columns=feat_def, fill_value=0)
            X_per = df.reindex(columns=feat_per, fill_value=0)

            # 6) Ensure numeric dtype
            X_lbl = X_lbl.apply(pd.to_numeric, errors="coerce").fillna(0)
            X_def = X_def.apply(pd.to_numeric, errors="coerce").fillna(0)
            X_per = X_per.apply(pd.to_numeric, errors="coerce").fillna(0)

            # 7) Compute probabilities and apply optimal thresholds
            prob_lbl = model_label.predict_proba(X_lbl)[:, 1]
            prob_def = model_defense.predict_proba(X_def)[:, 1]
            prob_per = model_persist.predict_proba(X_per)[:, 1]

            df["Pred_Label"]   = (prob_lbl >= self.optimal_thresholds["Label"]).astype(int)
            df["Pred_Defense"] = (prob_def >= self.optimal_thresholds["Defense"]).astype(int)
            df["Pred_Persist"] = (prob_per >= self.optimal_thresholds["Persistence"]).astype(int)

            # 8) If ground-truth available, update GUI metrics
            if {"Label", "Tactic"}.issubset(df.columns):
                metrics = {}
                metrics |= self.evaluatePredictions(
                    df["Label"].eq("Malicious").astype(int),
                    df["Pred_Label"], "Label"
                )
                metrics |= self.evaluatePredictions(
                    df["Tactic"].eq("Defense Evasion").astype(int),
                    df["Pred_Defense"], "Defense"
                )
                metrics |= self.evaluatePredictions(
                    df["Tactic"].eq("Persistence").astype(int),
                    df["Pred_Persist"], "Persistence"
                )
                self.updateMetricsDisplay(metrics)

            # 9) Save classified CSV
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.sModelOutputDir, f"classified_output_{ts}.csv")
            df.to_csv(out_path, index=False)
            messagebox.showinfo("Classification Complete",
                                f"Results saved to:\n{out_path}")

        except Exception as exc:
            messagebox.showerror("Classification error", str(exc))

    def evaluatePredictions(self, y_true, y_pred, tag):
        return {
            f"{tag} Accuracy":  f"{accuracy_score(y_true, y_pred):.4f}",
            f"{tag} Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
            f"{tag} Recall":    f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
            f"{tag} F1":        f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        }

    ###########################################################################
    #                       UPDATING
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, val in metrics.items():
            self.metricsList.insert("", "end", values=(metric, val))
                
    def updateFeatureDisplay(self,
                             lbl_feats, lbl_imps,
                             def_feats, def_imps,
                             per_feats, per_imps):
        """
        Populate the three tabs with feature importances for
        Label, Defense-Evasion, and Persistence models.
        """

        # clear all three trees
        for tv in self.featureTabsTabs.values():
            tv.delete(*tv.get_children())

        # Label tab
        for f, imp in sorted(zip(lbl_feats, lbl_imps), key=lambda x: x[1], reverse=True):
            self.featureTabsTabs["Label"].insert("", "end", values=(f, f"{imp:.4f}"))

        # Defense-Evasion tab
        for f, imp in sorted(zip(def_feats, def_imps), key=lambda x: x[1], reverse=True):
            self.featureTabsTabs["Defense"].insert("", "end", values=(f, f"{imp:.4f}"))

        # Persistence tab
        for f, imp in sorted(zip(per_feats, per_imps), key=lambda x: x[1], reverse=True):
            self.featureTabsTabs["Persistence"].insert("", "end", values=(f, f"{imp:.4f}"))

###########################################################################
#                       MAIN
###########################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
