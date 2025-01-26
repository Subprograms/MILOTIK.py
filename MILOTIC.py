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

        # User-specified paths
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

        self.selected_features = None

        self.setupUI()

    ###########################################################################
    #                           GUI
    ###########################################################################
    def setupUI(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        ttk.Label(frame, text="Hive Path:").grid(row=0, column=0, sticky='e')
        self.hivePathInput = ttk.Entry(frame, width=50)
        self.hivePathInput.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Set Hive Path", command=self.setHivePath).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Malicious Keys File:").grid(row=1, column=0, sticky='e')
        self.maliciousKeysInput = ttk.Entry(frame, width=50)
        self.maliciousKeysInput.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Set Malicious Keys", command=self.setMaliciousKeysPath).grid(row=1, column=2, padx=5)

        ttk.Label(frame, text="Tagged Keys File:").grid(row=2, column=0, sticky='e')
        self.taggedKeysInput = ttk.Entry(frame, width=50)
        self.taggedKeysInput.grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Set Tagged Keys", command=self.setTaggedKeysPath).grid(row=2, column=2, padx=5)

        ttk.Label(frame, text="Training Dataset (Optional):").grid(row=3, column=0, sticky='e')
        self.trainingDatasetInput = ttk.Entry(frame, width=50)
        self.trainingDatasetInput.grid(row=3, column=1, padx=5)
        ttk.Button(frame, text="Set Training Dataset", command=self.setTrainingDatasetPath).grid(row=3, column=2, padx=5)

        ttk.Label(frame, text="Raw Parsed CSV (Optional):").grid(row=4, column=0, sticky='e')
        self.rawParsedCsvInput = ttk.Entry(frame, width=50)
        self.rawParsedCsvInput.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Set Raw Parsed CSV", command=self.setRawParsedCsvPath).grid(row=4, column=2, padx=5)

        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=5, column=0, sticky='e')
        self.classifyCsvInput = ttk.Entry(frame, width=50)
        self.classifyCsvInput.grid(row=5, column=1, padx=5)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=5, column=2, padx=5)

        ttk.Label(frame, text="Label Model (Optional):").grid(row=6, column=0, sticky='e')
        self.labelModelInput = ttk.Entry(frame, width=50)
        self.labelModelInput.grid(row=6, column=1, padx=5)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=6, column=2, padx=5)

        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=7, column=0, sticky='e')
        self.tacticModelInput = ttk.Entry(frame, width=50)
        self.tacticModelInput.grid(row=7, column=1, padx=5)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=7, column=2, padx=5)

        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=8, column=0, sticky='e')
        self.persistenceModelInput = ttk.Entry(frame, width=50)
        self.persistenceModelInput.grid(row=8, column=1, padx=5)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=8, column=2, padx=5)

        ttk.Button(frame, text="Make Dataset", command=self.makeDataset).grid(row=9, column=1, pady=10)
        ttk.Button(frame, text="Start ML Process", command=self.executeMLProcess).grid(row=10, column=1, pady=10)

        self.metricsList = ttk.Treeview(frame, columns=("Metric", "Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value", text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value", width=500, anchor="w")
        self.metricsList.grid(row=11, column=0, columnspan=3, pady=10)

    ###########################################################################
    #                           PATH SETTERS
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
    #                        RAW PARSING & DATASET CREATION
    ###########################################################################
    def parseRegistry(self, sHivePath):
        """
        Parse the registry hive (raw). Return a DataFrame that includes
        'Key', 'Depth', 'Key Size', 'Subkey Count', 'Value Count', 'Name', 'Value', 'Type'.
        """
        xData = []
        subkey_counts = {}
        try:
            with ThreadPoolExecutor() as executor:
                xHive = RegistryHive(sHivePath)
                for xSubkey in xHive.recurse_subkeys():
                    sKeyPath = xSubkey.path
                    parent_path = '\\'.join(sKeyPath.split('\\')[:-1])
                    subkey_counts[parent_path] = subkey_counts.get(parent_path, 0) + 1

                    nDepth = sKeyPath.count('\\')
                    nKeySize = len(sKeyPath.encode('utf-8'))
                    nValueCount = len(xSubkey.values)
                    nSubkeyCount = subkey_counts.get(sKeyPath, 0)

                    for xValue in xSubkey.values:
                        xData.append({
                            "Key": sKeyPath,
                            "Depth": nDepth,
                            "Key Size": nKeySize,
                            "Subkey Count": nSubkeyCount,
                            "Value Count": nValueCount,
                            "Name": xValue.name,
                            "Value": str(xValue.value),
                            "Type": xValue.value_type
                        })
            return pd.DataFrame(xData)

        except Exception as e:
            messagebox.showerror("Error", f"Error parsing hive: {e}")
            return pd.DataFrame()

    def makeDataset(self):
        """
        - Parse registry data (raw).
        - Apply labels -> store in a raw/labeled CSV (with 'Key' for reference).
        - Then create a preprocessed DataFrame (still includes 'Key') for training dataset.
        """
        try:
            if not os.path.exists(self.sHivePath):
                raise FileNotFoundError("Hive path not found.")

            print("Parsing registry data...")
            df_raw = self.parseRegistry(self.sHivePath)

            print("Applying labels...")
            df_labeled = self.applyLabels(df_raw)

            # Save or append *raw-labeled* data to raw_parsed CSV
            if self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                self.appendToExistingCsv(df_labeled, self.sRawParsedCsvPath)
                print(f"Appended labeled raw data to existing: {self.sRawParsedCsvPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_raw_parsed = os.path.join(self.sModelOutputDir, f"raw_parsed_{ts}.csv")
                df_labeled.to_csv(new_raw_parsed, index=False)
                self.sRawParsedCsvPath = new_raw_parsed
                self.rawParsedCsvInput.delete(0, tk.END)
                self.rawParsedCsvInput.insert(0, new_raw_parsed)
                print(f"New raw parsed CSV created: {new_raw_parsed}")

            # Preprocess data for training, but keep 'Key' for reference
            print("Preprocessing data for training (but keep 'Key')...")
            df_preproc = self.preprocessData(df_labeled)

            # Save or append preprocessed data (with Key) to training dataset
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                self.appendToExistingCsv(df_preproc, self.sTrainingDatasetPath)
                print(f"Appended data to existing training dataset: {self.sTrainingDatasetPath}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_training_dataset_path = os.path.join(self.sModelOutputDir, f"training_dataset_{ts}.csv")
                df_preproc.to_csv(new_training_dataset_path, index=False)
                self.sTrainingDatasetPath = new_training_dataset_path
                self.trainingDatasetInput.delete(0, tk.END)
                self.trainingDatasetInput.insert(0, new_training_dataset_path)
                print(f"New training dataset created: {new_training_dataset_path}")
                messagebox.showinfo("Dataset Created", f"New training dataset created: {new_training_dataset_path}")

        except Exception as e:
            msg = f"Error in makeDataset: {e}"
            print(msg)
            messagebox.showerror("Error", msg)

    def appendToExistingCsv(self, new_df: pd.DataFrame, sAppendPath: str):
        try:
            if not sAppendPath or not os.path.exists(sAppendPath):
                new_df.to_csv(sAppendPath, index=False)
                print(f"Data saved to new CSV: {sAppendPath}")
                return

            existing_df = pd.read_csv(sAppendPath)
            if set(existing_df.columns) != set(new_df.columns):
                print("Column mismatch. Overwriting existing CSV with new data.")
                new_df.to_csv(sAppendPath, index=False)
                return

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(sAppendPath, index=False)
            print(f"Data appended to existing CSV: {sAppendPath}")

        except Exception as e:
            print(f"Error appending to CSV ({sAppendPath}): {e}")
            new_df.to_csv(sAppendPath, index=False)

    ###########################################################################
    #                           APPLY LABELS
    ###########################################################################
    def applyLabels(self, df):
        """Add 'Label' (Malicious/Benign) and 'Tactic' columns to raw data. Keep 'Key' etc."""
        try:
            if 'Key' not in df.columns:
                raise KeyError("The 'Key' column is missing from the DataFrame.")

            # Malicious file
            malicious_entries = []
            if self.sMaliciousKeysPath and os.path.exists(self.sMaliciousKeysPath):
                with open(self.sMaliciousKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": re.sub(r'\\+', r'\\', parts[0].strip()),
                            "Name": parts[1].strip() if len(parts) > 1 and parts[1].lower() != "none" else None,
                            "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts) > 2 and parts[2].lower() != "none" else None,
                            "Type": parts[3].strip() if len(parts) > 3 and parts[3].lower() != "none" else None
                        }
                        malicious_entries.append(entry)

            # Tagged file
            tagged_entries = []
            if self.sTaggedKeysPath and os.path.exists(self.sTaggedKeysPath):
                with open(self.sTaggedKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[\,\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": re.sub(r'\\+', r'\\', parts[0].strip()),
                            "Name": parts[1].strip() if len(parts) > 1 and parts[1].lower() != "none" else None,
                            "Value": re.sub(r'\\+', r'\\', parts[2].strip()) if len(parts) > 2 and parts[2].lower() != "none" else None,
                            "Type": parts[3].strip() if len(parts) > 3 and parts[3].lower() != "none" else None,
                            "Tactic": parts[4].strip() if len(parts) > 4 else "Persistence"
                        }
                        tagged_entries.append(entry)

            def is_malicious(row):
                row_key = re.sub(r'\\+', r'\\', str(row.get('Key', '')).lower())
                row_name = str(row.get('Name', '')).lower().strip()
                row_value = re.sub(r'\\+', r'\\', str(row.get('Value', '')).lower().strip())
                row_type = str(row.get('Type', '')).lower().strip()

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
                row_key = re.sub(r'\\+', r'\\', str(row.get('Key', '')).lower())
                row_name = str(row.get('Name', '')).lower().strip()
                row_value = re.sub(r'\\+', r'\\', str(row.get('Value', '')).lower().strip())
                row_type = str(row.get('Type', '')).lower().strip()

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

        except Exception as e:
            print(f"Error applying labels: {e}")
            raise RuntimeError(f"Error applying labels: {e}")

    ###########################################################################
    #                       PREPROCESSING FOR TRAINING
    ###########################################################################
    def preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        We keep 'Key' for reference in the final output, but we do not feed it to the model.
        The final returned DataFrame includes 'Key', 'Label', 'Tactic', plus processed columns.
        """
        if df.empty:
            print("No valid data to preprocess.")
            return pd.DataFrame()

        from sklearn.preprocessing import MinMaxScaler, RobustScaler

        xDf = df.copy()
        # Fill numeric NaNs
        xDf.fillna(xDf.select_dtypes(include=[np.number]).mean(), inplace=True)

        # We'll keep 'Key' so we can see it in the final CSV, but not for ML
        # We'll keep 'Name' as well if you want, or drop it:
        # xDf keeps them for reference in the final dataset

        # Basic encodings
        xDf['Path Category'] = xDf['Key'].apply(self.categorizePath)
        path_encoded = pd.get_dummies(xDf['Path Category'], prefix='PathCategory')
        xDf = pd.concat([xDf, path_encoded], axis=1)

        xDf['Type Group'] = xDf['Type'].apply(self.mapType)
        type_group_encoded = pd.get_dummies(xDf['Type Group'], prefix='TypeGroup')
        xDf = pd.concat([xDf, type_group_encoded], axis=1)

        xDf['Key Name Category'] = xDf['Name'].apply(self.categorizeKeyName)
        key_name_encoded = pd.get_dummies(xDf['Key Name Category'], prefix='KeyNameCategory')
        xDf = pd.concat([xDf, key_name_encoded], axis=1)

        # Convert 'Value' column to numeric measure
        xDf['Value Processed'] = xDf['Value'].apply(self.preprocessValue)

        # Scale numeric
        # We'll do it in-place on numeric columns. But keep 'Key','Name','Value','Label','Tactic' as is.
        # So we define numeric subset
        scaler_minmax = MinMaxScaler()
        minmax_cols = ['Depth', 'Value Count', 'Value Processed']
        xDf[minmax_cols] = scaler_minmax.fit_transform(xDf[minmax_cols])

        scaler_robust = RobustScaler()
        robust_cols = ['Key Size', 'Subkey Count']
        xDf[robust_cols] = scaler_robust.fit_transform(xDf[robust_cols])

        return xDf

    def categorizePath(self, path):
        if "Run" in path:
            return "Startup Path"
        elif "Services" in path:
            return "Service Path"
        elif "Internet Settings" in path:
            return "Network Path"
        return "Other Path"

    def mapType(self, value_type):
        type_map = {
            "String": ["REG_SZ", "REG_EXPAND_SZ", "REG_MULTI_SZ"],
            "Numeric": ["REG_DWORD", "REG_QWORD"],
            "Binary": ["REG_BINARY"],
            "Others": ["REG_NONE", "REG_LINK"]
        }
        for group, regtypes in type_map.items():
            if value_type in regtypes:
                return group
        return "Others"

    def categorizeKeyName(self, key_name):
        categories = {
            "Run Keys": ["Run", "RunOnce", "RunServices"],
            "Service Keys": ["ImageFileExecutionOptions", "AppInit_DLLs"],
            "Security and Configuration Keys": ["Policies", "Explorer"],
            "Internet and Network Keys": ["ProxyEnable", "ProxyServer"],
            "File Execution Keys": ["ShellExecuteHooks"]
        }
        for category, keywords in categories.items():
            if any(k in key_name for k in keywords):
                return category
        return "Other Keys"

    def preprocessValue(self, val):
        if isinstance(val, str):
            return len(val)
        return val

    ###########################################################################
    #                       EXECUTE ML PROCESS
    ###########################################################################
    def executeMLProcess(self):
        """
        If no classify CSV is set, uses the training dataset for classification 
        (testing scenario). Then trains/evaluates the models and does classification.
        """
        try:
            # No classify CSV => default to training dataset
            if not self.sClassifyCsvPath:
                print("No CSV to classify provided, using training dataset for classification test.")
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            print("Loading training dataset for ML...")
            df_train = pd.read_csv(self.sTrainingDatasetPath)

            # Train/Eval
            self.trainAndEvaluateModels(df_train)

            # Classification
            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")

        except Exception as e:
            messagebox.showerror("Error", f"Error in ML process: {e}")

    ###########################################################################
    #                   TRAIN AND EVALUATE MODELS
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        3 separate RandomForest models:
         - Label (Malicious vs Benign)
         - Defense Evasion Tactic
         - Persistence Tactic

        Shows accuracy, precision, recall, F1, and AUC. 
        If single-class only (i.e. all benign), force 30% to another class for demonstration.
        """
        try:
            if df.empty:
                raise ValueError("DataFrame is empty.")

            # The final CSV from makeDataset includes 'Key','Name','Value','Label','Tactic', plus processed columns
            # For ML, we exclude 'Key','Name','Value','Label','Tactic'
            drop_for_X = []
            for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                if c in df.columns:
                    drop_for_X.append(c)

            # Features
            X = df.drop(columns=drop_for_X, errors='ignore').copy()

            # Targets
            if 'Label' not in df.columns or 'Tactic' not in df.columns:
                raise ValueError("Missing 'Label' or 'Tactic' in dataset.")

            y_label = (df['Label'] == 'Malicious').astype(int)
            y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
            y_persistence = (df['Tactic'] == 'Persistence').astype(int)

            # Force multi-class
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE
            from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            if y_label.nunique() < 2:
                flip_n = int(len(y_label)*0.3)
                idx = np.random.choice(y_label.index, size=flip_n, replace=False)
                y_label.iloc[idx] = 1
                print("Forcing 30% Malicious for label model test.")

            if y_defense.nunique() < 2:
                flip_n = int(len(y_defense)*0.3)
                idx = np.random.choice(y_defense.index, size=flip_n, replace=False)
                y_defense.iloc[idx] = 1
                print("Forcing 30% Defense Evasion for tactic test.")

            if y_persistence.nunique() < 2:
                flip_n = int(len(y_persistence)*0.3)
                idx = np.random.choice(y_persistence.index, size=flip_n, replace=False)
                y_persistence.iloc[idx] = 1
                print("Forcing 30% Persistence for tactic test.")

            # RFE for label model to select features
            print("Performing RFE for feature selection (Label model base).")
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=base_model, n_features_to_select=10)
            rfe.fit(X, y_label)
            self.selected_features = X.columns[rfe.support_]
            print("Selected features:", list(self.selected_features))

            def grid_search_rf(Xp, yp):
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 10],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False]
                }
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                model = RandomForestClassifier(random_state=42)
                gs = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
                gs.fit(Xp, yp)
                return gs.best_estimator_

            # Evaluate final model with a train/test split
            def evaluate_model(model, Xd, yd, label_name=""):
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(Xd, yd, test_size=0.2,
                                                                    random_state=42, stratify=yd)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                try:
                    y_scores = model.predict_proba(X_test)[:, 1]
                except AttributeError:
                    y_scores = None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1v = f1_score(y_test, y_pred, zero_division=0)

                aucv = 0.0
                if y_scores is not None and len(np.unique(y_test))>1:
                    aucv = roc_auc_score(y_test, y_scores)

                print(f"{label_name} => Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1v:.4f}, AUC={aucv:.4f}")

                return {
                    f"{label_name} Accuracy": acc,
                    f"{label_name} Precision": prec,
                    f"{label_name} Recall": rec,
                    f"{label_name} F1": f1v,
                    f"{label_name} AUC": aucv
                }

            # Filter X to self.selected_features
            X_sel = X[self.selected_features].copy()

            # 1) Label Model
            print("Training Label Model.")
            label_model = grid_search_rf(X_sel, y_label)
            label_metrics = evaluate_model(label_model, X_sel, y_label, "Label Model")
            label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
            joblib.dump(label_model, label_model_path)
            self.sLabelModelPath = label_model_path

            # 2) Defense Evasion
            print("Training Defense Evasion Model.")
            defense_model = grid_search_rf(X_sel, y_defense)
            defense_metrics = evaluate_model(defense_model, X_sel, y_defense, "Defense Evasion Model")
            defense_model_path = os.path.join(self.sModelOutputDir, "defense_model.joblib")
            joblib.dump(defense_model, defense_model_path)
            self.sTacticModelPath = defense_model_path

            # 3) Persistence
            print("Training Persistence Model.")
            persistence_model = grid_search_rf(X_sel, y_persistence)
            persistence_metrics = evaluate_model(persistence_model, X_sel, y_persistence, "Persistence Model")
            persistence_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
            joblib.dump(persistence_model, persistence_model_path)
            self.sPersistenceModelPath = persistence_model_path

            # Merge metrics
            combined = {}
            combined.update(label_metrics)
            combined.update(defense_metrics)
            combined.update(persistence_metrics)

            # Convert to strings
            out_metrics = {k: f"{v:.4f}" for k,v in combined.items()}
            self.updateMetricsDisplay(out_metrics)

        except Exception as e:
            raise RuntimeError(f"Training error: {e}")

    ###########################################################################
    #                          CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path):
        """
        Classify a CSV using the three trained models.
        The CSV is assumed to be preprocessed data with 'Key' for reference, 
        plus the same columns used in training, so we can filter out 'Key' 
        from features. We check selected_features explicitly (len != 0).
        """
        try:
            df = pd.read_csv(csv_path)

            # Check if we have selected_features
            if self.selected_features is None or len(self.selected_features) == 0:
                raise ValueError("No selected_features found. Did you run training?")

            label_model = joblib.load(self.sLabelModelPath)
            defense_model = joblib.load(self.sTacticModelPath)
            persistence_model = joblib.load(self.sPersistenceModelPath)

            # For classification, remove the columns we don't want in X
            # We'll keep 'Key' in the output but not feed it to the model.
            drop_for_X = []
            for c in ['Key','Name','Value','Label','Tactic','Type','Type Group','Key Name Category','Path Category']:
                if c in df.columns:
                    drop_for_X.append(c)

            # Filter by selected_features from what's left
            X_all = df.drop(columns=drop_for_X, errors='ignore')
            X = X_all[self.selected_features].copy()

            # Predict label
            y_scores_label = label_model.predict_proba(X)[:, 1]
            y_pred_label = np.where(y_scores_label>=0.5, 'Malicious','Benign')

            # Predict tactic
            y_scores_defense = defense_model.predict_proba(X)[:, 1]
            y_pred_defense = np.where(y_scores_defense>=0.5,'Defense Evasion','None')

            y_scores_persist = persistence_model.predict_proba(X)[:,1]
            y_pred_persist = np.where(y_scores_persist>=0.5,'Persistence','None')

            # Final tactic
            df['Predicted Label'] = y_pred_label
            df['Predicted Tactic'] = np.where(y_pred_defense == 'Defense Evasion',
                                              'Defense Evasion',
                                              y_pred_persist)

            out_csv = os.path.join(self.sModelOutputDir,
                f"classified_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Classified output saved to: {out_csv}")
            messagebox.showinfo("Classification Complete", f"Classified output saved to: {out_csv}")

        except Exception as e:
            raise RuntimeError(f"Classification error: {e}")

    ###########################################################################
    #                       METRICS DISPLAY
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, value in metrics.items():
            self.metricsList.insert("", "end", values=(metric, value))

if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
