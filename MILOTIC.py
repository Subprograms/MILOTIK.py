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
        self.root.rowconfigure(0,  weight=1)

        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, sticky="nsew")
        for r in range(0, 13):
            frame.rowconfigure(r, weight=0)
        frame.rowconfigure(10, weight=1)   # metrics list grows
        frame.rowconfigure(11, weight=1)   # feature list grows
        frame.columnconfigure(3, weight=1) # tree notebook grows

        ttk.Label(frame, text="Hive Path:").grid(row=0, column=0, sticky="e")
        self.hivePathInput = ttk.Entry(frame, width=50)
        self.hivePathInput.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Set Hive Path", command=self.setHivePath).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Malicious Keys File:").grid(row=1, column=0, sticky="e")
        self.maliciousKeysInput = ttk.Entry(frame, width=50)
        self.maliciousKeysInput.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Set Malicious Keys", command=self.setMaliciousKeysPath).grid(row=1, column=2, padx=5)

        ttk.Label(frame, text="Tagged Keys File:").grid(row=2, column=0, sticky="e")
        self.taggedKeysInput = ttk.Entry(frame, width=50)
        self.taggedKeysInput.grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Set Tagged Keys", command=self.setTaggedKeysPath).grid(row=2, column=2, padx=5)

        ttk.Label(frame, text="Training Dataset (Optional):").grid(row=3, column=0, sticky="e")
        self.trainingDatasetInput = ttk.Entry(frame, width=50)
        self.trainingDatasetInput.grid(row=3, column=1, padx=5)
        ttk.Button(frame, text="Set Training Dataset", command=self.setTrainingDatasetPath).grid(row=3, column=2, padx=5)

        ttk.Label(frame, text="Raw Parsed CSV (Optional):").grid(row=4, column=0, sticky="e")
        self.rawParsedCsvInput = ttk.Entry(frame, width=50)
        self.rawParsedCsvInput.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Set Raw Parsed CSV", command=self.setRawParsedCsvPath).grid(row=4, column=2, padx=5)

        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=5, column=0, sticky="e")
        self.classifyCsvInput = ttk.Entry(frame, width=50)
        self.classifyCsvInput.grid(row=5, column=1, padx=5)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=5, column=2, padx=5)

        ttk.Label(frame, text="Label Model (Optional):").grid(row=6, column=0, sticky="e")
        self.labelModelInput = ttk.Entry(frame, width=50)
        self.labelModelInput.grid(row=6, column=1, padx=5)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=6, column=2, padx=5)

        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=7, column=0, sticky="e")
        self.tacticModelInput = ttk.Entry(frame, width=50)
        self.tacticModelInput.grid(row=7, column=1, padx=5)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=7, column=2, padx=5)

        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=8, column=0, sticky="e")
        self.persistenceModelInput = ttk.Entry(frame, width=50)
        self.persistenceModelInput.grid(row=8, column=1, padx=5)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=8, column=2, padx=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Make Dataset", command=self.makeDataset).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Start ML Process", command=self.executeMLProcess).pack(side="left", padx=5)

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
        self.featureList = ttk.Treeview(feature_frame, columns=("Feature", "Importance"), show="headings")
        self.featureList.heading("Feature", text="Feature")
        self.featureList.heading("Importance", text="Importance")
        self.featureList.column("Feature",    width=200, anchor="w")
        self.featureList.column("Importance", width=500, anchor="w")
        self.featureList.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(feature_frame, orient="vertical", command=self.featureList.yview).pack(side="right", fill="y")

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
    
    def _init_zoom_canvas(self, tag):
        """
        Replace the (tab, Label) pair created in setupUI with a scroll-zoom Canvas.
        Call this once per tag when you first draw that tree.
        """
        if tag in getattr(self, "_tree_canvases", {}):
            return                                        # already initialised

        lbl      = self.img_labels[tag]
        parent   = lbl.master                            # the Notebook tab
        lbl.destroy()                                    # remove the plain Label

        # ------- canvas + scrollbars -----------------------------------------
        c       = tk.Canvas(parent, bg="white")          # will hold the image
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

        # ------- zoom + pan bindings -----------------------------------------
        c.bind("<MouseWheel>", lambda e, t=tag: self._on_zoom(e, t))
        c.bind("<ButtonPress-1>",lambda e, canv=c: canv.scan_mark(e.x, e.y))
        c.bind("<B1-Motion>",    lambda e, canv=c: canv.scan_dragto(e.x, e.y, 1))

        # ------- stash --------------------------------------------------------
        self._tree_canvases = getattr(self, "_tree_canvases", {})
        self._tree_canvases[tag] = {
            "canvas"   : c,
            "pil_orig" : None,     # full-res Pillow image
            "tk_img"   : None,
            "scale"    : 1.0,
        }

    def _on_zoom(self, event, tag):
        """
        Mouse-wheel zoom – loss-less because we always resample from the
        stored full-resolution bitmap.
        """
        rec         = self._tree_canvases[tag]
        pil_full    = rec["pil_orig"]

        # wheel direction
        delta       = 1.1 if event.delta > 0 else (1/1.1)
        new_scale   = rec["scale"] * delta
        new_scale   = min(max(new_scale, 0.1), 5.0)      # clamp 10 % … 500 %
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

        # == basic cleaning --------------==
        for col in df.columns:
            df[col] = df[col].apply(clean_text)
        df.replace("", "0", inplace=True)
        df.fillna("0", inplace=True)

        # == optional row prune -------------=
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

        # == engineered cols --------------
        xdf["Path Category"]     = xdf["Key"].apply(self.categorizePath)
        xdf["Type Group"]        = xdf["Type"].apply(self.mapType)
        xdf["Key Name Category"] = xdf["Name"].apply(self.categorizeKeyName)
        xdf["Value Processed"]   = xdf["Value"].apply(self.preprocessValue)

        # == expected categories (column universe) ------=
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

        # == scale numeric fields ------------=
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
        If all three model paths *and* a CSV-to-classify are supplied,
        skip training and go straight to classification.
        Otherwise run the full train-->-classify pipeline.
        """
        try:
            # ----------  fast-path: pre-trained models + CSV  ----------
            models_ready = all([
                self.sLabelModelPath       and os.path.exists(self.sLabelModelPath),
                self.sTacticModelPath      and os.path.exists(self.sTacticModelPath),
                self.sPersistenceModelPath and os.path.exists(self.sPersistenceModelPath)
            ])
            classify_ready = (
                self.sClassifyCsvPath and os.path.exists(self.sClassifyCsvPath)
            )

            if models_ready and classify_ready:
                print("[ML] Found three models + CSV -> skipping training.")
                self.classifyCsv(self.sClassifyCsvPath)
                messagebox.showinfo("ML Process Complete", "Classification finished!")
                return

            # ----------  build / load training dataframe  ---------------
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                print("[ML] Using supplied training dataset.")
                df_raw = self.safe_read_csv(self.sTrainingDatasetPath)

                # If the file is an *unprocessed* registry dump, convert it now
                if {"Key", "Name", "Type"}.issubset(df_raw.columns):
                    df_raw = self.clean_dataframe(df_raw, drop_all_zero_rows=True,
                                                  preserve_labels=True)
                    df_raw = self.preprocessData(df_raw)
                    df_raw = df_raw[self.selectTrainingColumns(df_raw)]

            elif self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                print("[ML] Pre-processing raw-parsed CSV …")
                tmp = self.safe_read_csv(self.sRawParsedCsvPath)
                tmp = self.clean_dataframe(tmp, drop_all_zero_rows=True,
                                           preserve_labels=True)
                df_raw = self.preprocessData(tmp)
                df_raw = df_raw[self.selectTrainingColumns(df_raw)]

            else:
                raise FileNotFoundError("No training dataset or raw-parsed CSV found.")

            # ----------  tidy columns  ----------------------------------
            if df_raw.columns.duplicated().any():
                df_raw = df_raw.groupby(axis=1, level=0).first()

            training_columns = {
                "Key", "Label", "Tactic",
                "Depth", "Key Size", "Subkey Count",
                "Value Count", "Value Processed",
                *[f"PathCategory_{c}" for c in
                  ["Startup Path", "Service Path", "Network Path", "Other Path"]],
                *[f"TypeGroup_{g}" for g in
                  ["String", "Numeric", "Binary", "Others"]],
                *[f"KeyNameCategory_{k}" for k in [
                    "Run Keys", "Service Keys", "Security and Configuration Keys",
                    "Internet and Network Keys", "File Execution Keys", "Other Keys"]]
            }
            df_raw = df_raw[[c for c in df_raw.columns if c in training_columns]]
            df_raw = self.clean_dataframe(df_raw, drop_all_zero_rows=True,
                                          preserve_labels=True)

            # ----------  choose CSV to classify if user omitted it  -----
            if not classify_ready:
                self.sClassifyCsvPath = self.sTrainingDatasetPath
                print("[ML] No classify CSV provided -> will classify training set.")

            # ----------  train -> evaluate -> save models  ----------------
            self.trainAndEvaluateModels(df_raw)

            # ----------  classify the chosen CSV  -----------------------
            self.classifyCsv(self.sClassifyCsvPath)
            messagebox.showinfo("ML Process Complete",
                                "Training + classification finished!")

        except Exception as exc:
            # Bubble up any problem in a GUI dialog
            messagebox.showerror("Error in ML process", str(exc))

    ###########################################################################
    #            TRAIN AND EVALUATE MODELS
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        RFE -> top-N features, currently at 10
        train three BRF models
        evaluate & push metrics to GUI
        dump .joblib models
        export a couple of trees to .PNG if GraphViz is present
        """
        if df.empty:
            raise ValueError("Training dataset is empty")

        for c in ["Depth", "Key Size", "Subkey Count", "Value Count", "Value Processed"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        non_feat = [
            "Key", "Name", "Value", "Label", "Tactic",
            "Type", "Type Group", "Key Name Category", "Path Category"
        ]
        X_all = df.drop(columns=[c for c in non_feat if c in df], errors="ignore")
        X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)

        y_lbl = (df["Label"] == "Malicious").astype(int)
        y_def = (df["Tactic"] == "Defense Evasion").astype(int)
        y_per = (df["Tactic"] == "Persistence").astype(int)

        missing = [n for n, y in [("Label", y_lbl), ("Defense-Evasion", y_def), ("Persistence", y_per)] if y.nunique() < 2]
        if missing:
            raise ValueError(f"Training set needs positives for: {', '.join(missing)}")

        X_tr, X_te, y_lbl_tr, y_lbl_te = train_test_split(
            X_all, y_lbl, test_size=0.2, stratify=y_lbl, random_state=42
        )
        y_def_tr, y_def_te = y_def.loc[X_tr.index], y_def.loc[X_te.index]
        y_per_tr, y_per_te = y_per.loc[X_tr.index], y_per.loc[X_te.index]

        rfe = RFE(
            BalancedRandomForestClassifier(
                sampling_strategy="all",
                replacement=True,
                bootstrap=False,
                n_estimators=100,
                random_state=42,
            ),
            n_features_to_select=10,
        )
        rfe.fit(X_tr, y_lbl_tr)
        self.selected_features = X_tr.columns[rfe.support_]

        X_tr_sel = X_tr[self.selected_features]
        X_te_sel = X_te[self.selected_features]

        def build_model():
            return BalancedRandomForestClassifier(
                sampling_strategy="all",
                replacement=True,
                bootstrap=False,
                n_estimators=400,
                max_depth=None,
                random_state=42,
            )

        label_model = build_model().fit(X_tr_sel, y_lbl_tr)
        defense_model = build_model().fit(X_tr_sel, y_def_tr)
        persistence_model = build_model().fit(X_tr_sel, y_per_tr)

        self.X_tr_last = X_tr_sel
        self.y_tr_lbl  = y_lbl_tr
        self.y_tr_def  = y_def_tr
        self.y_tr_per  = y_per_tr
        
        self._show_forest_tree(label_model,   "Label", list(self.selected_features), self.X_tr_last, self.y_tr_lbl)
        self._show_forest_tree(defense_model, "Defense", list(self.selected_features), self.X_tr_last, self.y_tr_def)
        self._show_forest_tree(persistence_model, "Persistence", list(self.selected_features), self.X_tr_last, self.y_tr_per)

        def metrics(m, X, y):
            y_pred = m.predict(X)
            y_prob = m.predict_proba(X)[:, 1]
            return {
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1": f1_score(y, y_pred, zero_division=0),
                "AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else 0.0,
            }

        all_metrics = {}
        for tag, model, X, y in [
            ("Label", label_model, X_te_sel, y_lbl_te),
            ("Defense", defense_model, X_te_sel, y_def_te),
            ("Persistence", persistence_model, X_te_sel, y_per_te),
        ]:
            for k, v in metrics(model, X, y).items():
                all_metrics[f"{tag} {k}"] = f"{v:.4f}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sLabelModelPath = os.path.join(self.sModelOutputDir, f"label_model_{ts}.joblib")
        self.sTacticModelPath = os.path.join(self.sModelOutputDir, f"defense_model_{ts}.joblib")
        self.sPersistenceModelPath = os.path.join(self.sModelOutputDir, f"persistence_model_{ts}.joblib")
        joblib.dump(label_model, self.sLabelModelPath)
        joblib.dump(defense_model, self.sTacticModelPath)
        joblib.dump(persistence_model, self.sPersistenceModelPath)

        with open(os.path.join(self.sModelOutputDir, "selected_features.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.selected_features))

        self.updateMetricsDisplay(all_metrics)
        self.updateFeatureDisplay(label_model.feature_importances_, self.selected_features)

    # ----------------------------------------------------------------------
    # 2.  Grid-search helpers
    # ----------------------------------------------------------------------           
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

    def _show_forest_tree(self, forest, tag: str, feature_names, X_train, y_train):
        scores = [est.score(X_train, y_train) for est in forest.estimators_]
        best_estimator = forest.estimators_[int(np.argmax(scores))]

        rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(24, 16), dpi=600)
        tree.plot_tree(
            best_estimator,
            feature_names=None,  # Avoids warning from sklearn
            class_names=["Benign/Pure", "Positive"],
            filled=True, rounded=True,
            impurity=False, proportion=False,
            ax=ax
        )
        ax.axis("off")
        fig.tight_layout(pad=0.3)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_full = Image.open(buf)

        self._init_zoom_canvas(tag)
        rec = self._tree_canvases[tag]
        canv = rec["canvas"]
        rec["pil_orig"] = pil_full

        base_zoom = {
            "Label": 2.5,
            "Defense": 0.5,
            "Persistence": 0.1,
        }.get(tag, 1.0)

        canv.update_idletasks()
        fit_scale = 1.0 if canv.winfo_width() <= 1 else canv.winfo_width() / pil_full.width
        scale = base_zoom * fit_scale

        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        preview = pil_full.resize(
            (max(1, int(pil_full.width * scale)), max(1, int(pil_full.height * scale))),
            resample
        )
        tk_img = ImageTk.PhotoImage(preview)
        rec["tk_img"] = tk_img
        rec["scale"] = scale

        canv.delete("all")
        canv.create_image(0, 0, anchor="nw", image=tk_img)
        canv.config(scrollregion=(0, 0, preview.width, preview.height))
    
    ###########################################################################
    #                    CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path: str):
        try:
            df = self.safe_read_csv(csv_path)

            if {"Key", "Name", "Type"}.issubset(df.columns):
                df = self.clean_dataframe(df, drop_all_zero_rows=True, preserve_labels=True)
                df = self.preprocessData(df)
                df = df[self.selectTrainingColumns(df)]
            else:
                df = self.clean_dataframe(df, drop_all_zero_rows=True, preserve_labels=True)

            model_label   = joblib.load(self.sLabelModelPath)
            model_defense = joblib.load(self.sTacticModelPath)
            model_persist = joblib.load(self.sPersistenceModelPath)

            if self.selected_features is None or len(self.selected_features) == 0:
                self.selected_features = list(model_label.feature_names_in_)

            for col in self.selected_features:
                if col not in df.columns:
                    df[col] = 0

            X_df = (
                df[self.selected_features]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )

            df["Pred_Label"]   = model_label.predict(X_df)
            df["Pred_Defense"] = model_defense.predict(X_df)
            df["Pred_Persist"] = model_persist.predict(X_df)

            metrics = {}
            if {"Label", "Tactic"}.issubset(df.columns):
                metrics |= self._evaluate_predictions(
                    df["Label"].eq("Malicious").astype(int), df["Pred_Label"], "Label")
                metrics |= self._evaluate_predictions(
                    df["Tactic"].eq("Defense Evasion").astype(int), df["Pred_Defense"], "Defense")
                metrics |= self._evaluate_predictions(
                    df["Tactic"].eq("Persistence").astype(int), df["Pred_Persist"], "Persistence")
                self.updateMetricsDisplay(metrics)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.sModelOutputDir, f"classified_output_{ts}.csv")
            df.to_csv(out_path, index=False)
            print(f"[ML] Saved classified output -> {out_path}")
            messagebox.showinfo("Classification Complete", f"Results saved to:\n{out_path}")

        except Exception as exc:
            messagebox.showerror("Classification error", str(exc))
            
    def _evaluate_predictions(self, y_true, y_pred, tag):
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
