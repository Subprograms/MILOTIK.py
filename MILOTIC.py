import os
import tkinter as tk
import pandas as pd
import numpy as np
import joblib
import re

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

class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("750x700")

        # Paths
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

        # State
        self.selected_features = None

        self.setupUI()

    ###########################################################################
    #                          GUI Setup
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
    #                          PATH SETTERS
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
    #                      MAKE DATASET
    ###########################################################################
    def parseRegistry(self, hive_path):
        """Returns a raw DataFrame from the hive with 'Key','Depth','Key Size',..."""
        xData = []
        subkey_counts = {}
        try:
            with ThreadPoolExecutor() as executor:
                hive = RegistryHive(hive_path)
                for subkey in hive.recurse_subkeys():
                    sKeyPath = subkey.path
                    parent_path = '\\'.join(sKeyPath.split('\\')[:-1])
                    subkey_counts[parent_path] = subkey_counts.get(parent_path, 0) + 1

                    nDepth = sKeyPath.count('\\')
                    nKeySize = len(sKeyPath.encode('utf-8'))
                    nValueCount = len(subkey.values)
                    nSubkeyCount = subkey_counts.get(sKeyPath, 0)

                    for val in subkey.values:
                        xData.append({
                            "Key": sKeyPath,
                            "Depth": nDepth,
                            "Key Size": nKeySize,
                            "Subkey Count": nSubkeyCount,
                            "Value Count": nValueCount,
                            "Name": val.name,
                            "Value": str(val.value),
                            "Type": val.value_type
                        })
            return pd.DataFrame(xData)
        except Exception as e:
            messagebox.showerror("Error", f"Error parsing hive: {e}")
            return pd.DataFrame()

    def makeDataset(self):
        """
        1) parse + label => raw-labeled
        2) save/append to raw_parsed CSV
        3) preprocess => save to training dataset
        """
        try:
            if not os.path.exists(self.sHivePath):
                raise FileNotFoundError("Hive path not found.")

            print("Parsing registry data...")
            df_raw = self.parseRegistry(self.sHivePath)

            print("Applying labels...")
            df_labeled = self.applyLabels(df_raw)

            # Save or append the labeled raw data (with Key)
            if self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                self.appendToExistingCsv(df_labeled, self.sRawParsedCsvPath)
                print(f"Appended labeled raw data to: {self.sRawParsedCsvPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_raw_csv = os.path.join(self.sModelOutputDir, f"raw_parsed_{ts}.csv")
                df_labeled.to_csv(new_raw_csv, index=False)
                self.sRawParsedCsvPath = new_raw_csv
                self.rawParsedCsvInput.delete(0, tk.END)
                self.rawParsedCsvInput.insert(0, new_raw_csv)
                print(f"Created new raw parsed CSV: {new_raw_csv}")

            print("Preprocessing data for training (keeping 'Key' column for reference)...")
            df_preproc = self.preprocessData(df_labeled)

            # Save or append to training dataset
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                self.appendToExistingCsv(df_preproc, self.sTrainingDatasetPath)
                print(f"Appended preprocessed data to training dataset: {self.sTrainingDatasetPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_train_csv = os.path.join(self.sModelOutputDir, f"training_dataset_{ts}.csv")
                df_preproc.to_csv(new_train_csv, index=False)
                self.sTrainingDatasetPath = new_train_csv
                self.trainingDatasetInput.delete(0, tk.END)
                self.trainingDatasetInput.insert(0, new_train_csv)
                print(f"Created new training dataset: {new_train_csv}")
                messagebox.showinfo("Dataset Created", f"New training dataset created: {new_train_csv}")

        except Exception as e:
            msg = f"Error in makeDataset: {e}"
            print(msg)
            messagebox.showerror("Error", msg)

    def appendToExistingCsv(self, new_df: pd.DataFrame, csv_path: str):
        """Append or overwrite, preserving columns. Keep 'Key' if it exists."""
        try:
            if not csv_path or not os.path.exists(csv_path):
                new_df.to_csv(csv_path, index=False)
                print(f"Data saved to new CSV: {csv_path}")
                return

            existing_df = pd.read_csv(csv_path)
            if set(existing_df.columns) != set(new_df.columns):
                print("Column mismatch. Overwriting existing CSV with new data.")
                new_df.to_csv(csv_path, index=False)
                return

            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            print(f"Appended data to existing CSV: {csv_path}")

        except Exception as ex:
            print(f"Error appending to CSV ({csv_path}): {ex}")
            new_df.to_csv(csv_path, index=False)

    ###########################################################################
    #                      APPLY LABELS
    ###########################################################################
    def applyLabels(self, df):
        """Add 'Label' and 'Tactic', keep 'Key', etc."""
        try:
            if 'Key' not in df.columns:
                raise KeyError("DataFrame lacks 'Key' column.")

            # malicious
            malicious_entries = []
            if self.sMaliciousKeysPath and os.path.exists(self.sMaliciousKeysPath):
                with open(self.sMaliciousKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": re.sub(r'\\+', r'\\', parts[0].strip()),
                            "Name": parts[1].strip() if len(parts)>1 and parts[1].lower() != "none" else None,
                            "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts)>2 and parts[2].lower() != "none" else None,
                            "Type": parts[3].strip() if len(parts)>3 and parts[3].lower() != "none" else None
                        }
                        malicious_entries.append(entry)

            # tagged
            tagged_entries = []
            if self.sTaggedKeysPath and os.path.exists(self.sTaggedKeysPath):
                with open(self.sTaggedKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[\,\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": re.sub(r'\\+', r'\\', parts[0].strip()),
                            "Name": parts[1].strip() if len(parts)>1 and parts[1].lower() != "none" else None,
                            "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts)>2 and parts[2].lower() != "none" else None,
                            "Type": parts[3].strip() if len(parts)>3 and parts[3].lower() != "none" else None,
                            "Tactic": parts[4].strip() if len(parts)>4 else "Persistence"
                        }
                        tagged_entries.append(entry)

            def is_malicious(row):
                row_key = re.sub(r'\\+', r'\\', str(row['Key']).lower())
                row_name = str(row['Name']).lower().strip()
                row_value = re.sub(r'\\+', r'\\', str(row['Value']).lower().strip())
                row_type = str(row['Type']).lower().strip()
                for e in malicious_entries:
                    ekey_last = e['Key'].strip().split('\\')[-1].lower()
                    row_key_last = row_key.split('\\')[-1]
                    if row_key_last != ekey_last:
                        continue
                    if e['Name'] and row_name != e['Name'].lower():
                        continue
                    if e['Value'] and row_value != e['Value'].lower():
                        continue
                    if e['Type'] and row_type != e['Type'].lower():
                        continue
                    return 'Malicious'
                return 'Benign'

            def assign_tactic(row):
                row_key = re.sub(r'\\+', r'\\', str(row['Key']).lower())
                row_name = str(row['Name']).lower().strip()
                row_value = re.sub(r'\\+', r'\\', str(row['Value']).lower().strip())
                row_type = str(row['Type']).lower().strip()
                for e in tagged_entries:
                    ekey_last = e['Key'].strip().split('\\')[-1].lower()
                    row_key_last = row_key.split('\\')[-1]
                    if row_key_last != ekey_last:
                        continue
                    if e['Name'] and row_name != e['Name'].lower():
                        continue
                    if e['Value'] and row_value != e['Value'].lower():
                        continue
                    if e['Type'] and row_type != e['Type'].lower():
                        continue
                    return e['Tactic']
                return 'None'

            df['Label'] = df.apply(is_malicious, axis=1)
            df['Tactic'] = df.apply(assign_tactic, axis=1)
            return df
        except Exception as ex:
            print(f"Error applying labels: {ex}")
            raise RuntimeError(f"Error applying labels: {ex}")

    ###########################################################################
    #                 PREPROCESS (BUT KEEP 'Key')
    ###########################################################################
    def preprocessData(self, df):
        """Encodes, scales, but keeps 'Key' in the final DataFrame for reference."""
        if df.empty:
            print("No data to preprocess.")
            return pd.DataFrame()

        from sklearn.preprocessing import MinMaxScaler, RobustScaler

        xDf = df.copy()
        xDf.fillna(xDf.select_dtypes(include=[np.number]).mean(), inplace=True)

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
        scaler_minmax = MinMaxScaler()
        minmax_cols = ['Depth','Value Count','Value Processed']
        xDf[minmax_cols] = scaler_minmax.fit_transform(xDf[minmax_cols])

        scaler_robust = RobustScaler()
        robust_cols = ['Key Size','Subkey Count']
        xDf[robust_cols] = scaler_robust.fit_transform(xDf[robust_cols])

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
            "String": ["REG_SZ", "REG_EXPAND_SZ","REG_MULTI_SZ"],
            "Numeric": ["REG_DWORD","REG_QWORD"],
            "Binary": ["REG_BINARY"],
            "Others": ["REG_NONE","REG_LINK"]
        }
        for g,vals in type_map.items():
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
        for cat, keys in categories.items():
            if any(k in kn for k in keys):
                return cat
        return "Other Keys"

    def preprocessValue(self, v):
        if isinstance(v, str):
            return len(v)
        return v

    ###########################################################################
    #                    EXECUTE ML PROCESS
    ###########################################################################
    def executeMLProcess(self):
        """
        If no CSV to classify => use training dataset for classification test.
        Then train->evaluate->classify.
        """
        try:
            if not self.sClassifyCsvPath:
                print("No classify CSV specified, defaulting to training dataset for classification test.")
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            df_train = pd.read_csv(self.sTrainingDatasetPath)
            self.trainAndEvaluateModels(df_train)

            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")
        except Exception as ex:
            messagebox.showerror("Error", f"Error in ML process: {ex}")

    ###########################################################################
    #                   TRAIN & EVALUATE
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        3 RandomForest models: Label, Defense Evasion, Persistence.
        Make 30% of single-class entries to another class (i.e. all benign)
        Show Acc, Prec, Rec, F1, AUC. Save models.
        'Key' is kept in df but excluded from the model features in ML execution.
        """
        try:
            if df.empty:
                raise ValueError("DataFrame is empty")

            # Exclude 'Key','Name','Value','Label','Tactic','Type', etc from features
            exclude_cols = []
            for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                if c in df.columns:
                    exclude_cols.append(c)

            X_all = df.drop(columns=exclude_cols, errors='ignore')
            if 'Label' not in df.columns or 'Tactic' not in df.columns:
                raise ValueError("Missing 'Label' or 'Tactic' columns.")

            y_label = (df['Label'] == 'Malicious').astype(int)
            y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
            y_persistence = (df['Tactic'] == 'Persistence').astype(int)

            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE
            from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            # Force multi-class if single
            if y_label.nunique()<2:
                flip_n = int(len(y_label)*0.3)
                idx = np.random.choice(y_label.index, size=flip_n, replace=False)
                y_label.iloc[idx] = 1
                print("Forced ~30% Malicious for label model test.")

            if y_defense.nunique()<2:
                flip_n = int(len(y_defense)*0.3)
                idx = np.random.choice(y_defense.index, size=flip_n, replace=False)
                y_defense.iloc[idx] = 1
                print("Forced ~30% Defense Evasion for tactic test.")

            if y_persistence.nunique()<2:
                flip_n = int(len(y_persistence)*0.3)
                idx = np.random.choice(y_persistence.index, size=flip_n, replace=False)
                y_persistence.iloc[idx] = 1
                print("Forced ~30% Persistence for tactic test.")

            print("Performing RFE for feature selection with Label model.")
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=base_model, n_features_to_select=10)
            rfe.fit(X_all, y_label)
            self.selected_features = X_all.columns[rfe.support_]
            print("Selected features:", list(self.selected_features))

            def grid_search_rf(Xp, yp):
                param_grid = {
                    'n_estimators':[50,100],
                    'max_depth':[None,10],
                    'min_samples_split':[2,10],
                    'min_samples_leaf':[1,2],
                    'bootstrap':[True,False]
                }
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                model = RandomForestClassifier(random_state=42)
                gs = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
                gs.fit(Xp, yp)
                return gs.best_estimator_

            def evaluate_model(model, Xd, yd, label_name=""):
                from sklearn.model_selection import train_test_split
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

                print(f"{label_name} => Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1v:.4f}, AUC: {aucv:.4f}")
                return {
                    f"{label_name} Accuracy":acc,
                    f"{label_name} Precision":prec,
                    f"{label_name} Recall":rec,
                    f"{label_name} F1":f1v,
                    f"{label_name} AUC":aucv
                }

            X_sel = X_all[self.selected_features].copy()

            print("Training Label Model.")
            label_model = grid_search_rf(X_sel, y_label)
            label_metrics = evaluate_model(label_model, X_sel, y_label, "Label Model")
            label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
            joblib.dump(label_model, label_model_path)
            self.sLabelModelPath = label_model_path

            print("Training Defense Evasion Model.")
            defense_model = grid_search_rf(X_sel, y_defense)
            defense_metrics = evaluate_model(defense_model, X_sel, y_defense, "Defense Evasion Model")
            defense_model_path = os.path.join(self.sModelOutputDir, "defense_model.joblib")
            joblib.dump(defense_model, defense_model_path)
            self.sTacticModelPath = defense_model_path

            print("Training Persistence Model.")
            persistence_model = grid_search_rf(X_sel, y_persistence)
            persist_metrics = evaluate_model(persistence_model, X_sel, y_persistence, "Persistence Model")
            persistence_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
            joblib.dump(persistence_model, persistence_model_path)
            self.sPersistenceModelPath = persistence_model_path

            merged = {}
            merged.update(label_metrics)
            merged.update(defense_metrics)
            merged.update(persist_metrics)

            out_metrics = {k: f"{v:.4f}" for k,v in merged.items()}
            self.updateMetricsDisplay(out_metrics)

        except Exception as ex:
            raise RuntimeError(f"Training error: {ex}")

    ###########################################################################
    #                     CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path):
        """
        Always ensure final output has 'Key' column. 
        If it doesn't exist in the input, create an empty 'Key' column.
        We exclude 'Key' from features. 
        """
        try:
            df = pd.read_csv(csv_path)

            # Ensure we keep or add 'Key' so final CSV has it
            if 'Key' not in df.columns:
                df['Key'] = ""  # or "NoKeyProvided"

            if self.selected_features is None or len(self.selected_features) == 0:
                raise ValueError("No selected_features found. Did you run the training process?")

            # Load models
            label_model = joblib.load(self.sLabelModelPath)
            defense_model = joblib.load(self.sTacticModelPath)
            persistence_model = joblib.load(self.sPersistenceModelPath)

            # Exclude columns that aren't in selected_features
            exclude_cols = []
            for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                if c in df.columns:
                    exclude_cols.append(c)
            X_all = df.drop(columns=exclude_cols, errors='ignore')

            # Filter to the features we used
            X = X_all[self.selected_features].copy()

            # Label
            y_scores_label = label_model.predict_proba(X)[:,1]
            y_pred_label = np.where(y_scores_label>=0.5, 'Malicious','Benign')

            # Defense Evasion
            y_scores_defense = defense_model.predict_proba(X)[:,1]
            y_pred_defense = np.where(y_scores_defense>=0.5,'Defense Evasion','None')

            # Persistence
            y_scores_persist = persistence_model.predict_proba(X)[:,1]
            y_pred_persist = np.where(y_scores_persist>=0.5,'Persistence','None')

            df['Predicted Label'] = y_pred_label
            df['Predicted Tactic'] = np.where(y_pred_defense=='Defense Evasion','Defense Evasion',y_pred_persist)

            out_csv = os.path.join(self.sModelOutputDir,
                                   f"classified_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Classified output saved to: {out_csv}")
            messagebox.showinfo("Classification Complete", f"Classified output saved to: {out_csv}")

        except Exception as ex:
            raise RuntimeError(f"Classification error: {ex}")

    ###########################################################################
    #                     METRICS DISPLAY
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        """Show metrics in the Treeview."""
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, val in metrics.items():
            self.metricsList.insert("", "end", values=(metric, val))

if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
