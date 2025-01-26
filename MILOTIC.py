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
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score, make_scorer
)
from sklearn.feature_selection import RFE

class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("750x700")

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

        self.model_loaded = False
        self.selected_features = None

        self.setupUI()

    ###########################################################################
    #                            UI SETUP
    ###########################################################################
    def setupUI(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Row 0: Hive Path
        ttk.Label(frame, text="Hive Path:").grid(row=0, column=0, sticky='e')
        self.hivePathInput = ttk.Entry(frame, width=50)
        self.hivePathInput.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Set Hive Path", command=self.setHivePath).grid(row=0, column=2, padx=5)

        # Row 1: Malicious Keys
        ttk.Label(frame, text="Malicious Keys File:").grid(row=1, column=0, sticky='e')
        self.maliciousKeysInput = ttk.Entry(frame, width=50)
        self.maliciousKeysInput.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Set Malicious Keys", command=self.setMaliciousKeysPath).grid(row=1, column=2, padx=5)

        # Row 2: Tagged Keys
        ttk.Label(frame, text="Tagged Keys File:").grid(row=2, column=0, sticky='e')
        self.taggedKeysInput = ttk.Entry(frame, width=50)
        self.taggedKeysInput.grid(row=2, column=1, padx=5)
        ttk.Button(frame, text="Set Tagged Keys", command=self.setTaggedKeysPath).grid(row=2, column=2, padx=5)

        # Row 3: Training Dataset
        ttk.Label(frame, text="Training Dataset (Optional):").grid(row=3, column=0, sticky='e')
        self.trainingDatasetInput = ttk.Entry(frame, width=50)
        self.trainingDatasetInput.grid(row=3, column=1, padx=5)
        ttk.Button(frame, text="Set Training Dataset", command=self.setTrainingDatasetPath).grid(row=3, column=2, padx=5)

        # Row 4: Raw Parsed CSV
        ttk.Label(frame, text="Raw Parsed CSV (Optional):").grid(row=4, column=0, sticky='e')
        self.rawParsedCsvInput = ttk.Entry(frame, width=50)
        self.rawParsedCsvInput.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Set Raw Parsed CSV", command=self.setRawParsedCsvPath).grid(row=4, column=2, padx=5)

        # Row 5: CSV to Classify
        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=5, column=0, sticky='e')
        self.classifyCsvInput = ttk.Entry(frame, width=50)
        self.classifyCsvInput.grid(row=5, column=1, padx=5)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=5, column=2, padx=5)

        # Row 6: Label Model
        ttk.Label(frame, text="Label Model (Optional):").grid(row=6, column=0, sticky='e')
        self.labelModelInput = ttk.Entry(frame, width=50)
        self.labelModelInput.grid(row=6, column=1, padx=5)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=6, column=2, padx=5)

        # Row 7: Defense Evasion Model
        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=7, column=0, sticky='e')
        self.tacticModelInput = ttk.Entry(frame, width=50)
        self.tacticModelInput.grid(row=7, column=1, padx=5)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=7, column=2, padx=5)

        # Row 8: Persistence Model
        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=8, column=0, sticky='e')
        self.persistenceModelInput = ttk.Entry(frame, width=50)
        self.persistenceModelInput.grid(row=8, column=1, padx=5)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=8, column=2, padx=5)

        # Row 9/10: Buttons
        ttk.Button(frame, text="Make Dataset", command=self.makeDataset).grid(row=9, column=1, pady=10)
        ttk.Button(frame, text="Start ML Process", command=self.executeMLProcess).grid(row=10, column=1, pady=10)

        # Row 11: Metrics Tree
        self.metricsList = ttk.Treeview(frame, columns=("Metric", "Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value", text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value", width=500, anchor="w")
        self.metricsList.grid(row=11, column=0, columnspan=3, pady=10)

    ###########################################################################
    #                            SETTERS
    ###########################################################################
    def setHivePath(self):
        self.sHivePath = self.hivePathInput.get().strip()
        messagebox.showinfo("Path Set", f"Hive path set to: {self.sHivePath}")

    def setMaliciousKeysPath(self):
        self.sMaliciousKeysPath = self.maliciousKeysInput.get().strip()
        messagebox.showinfo("Path Set", f"Malicious keys file set to: {self.sMaliciousKeysPath}")

    def setRawParsedCsvPath(self):
        self.sRawParsedCsvPath = self.rawParsedCsvInput.get().strip()
        messagebox.showinfo("Path Set", f"Raw parsed CSV set to: {self.sRawParsedCsvPath}")

    def setTaggedKeysPath(self):
        self.sTaggedKeysPath = self.taggedKeysInput.get().strip()
        messagebox.showinfo("Path Set", f"Tagged keys file set to: {self.sTaggedKeysPath}")

    def setTrainingDatasetPath(self):
        self.sTrainingDatasetPath = self.trainingDatasetInput.get().strip()
        messagebox.showinfo("Path Set", f"Training dataset path set to: {self.sTrainingDatasetPath}")

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
    #                     CREATE / APPEND DATASET
    ###########################################################################
    def parseRegistry(self, sHivePath):
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
        try:
            if not os.path.exists(self.sHivePath):
                raise FileNotFoundError("Hive path not found.")

            print("Parsing registry data...")
            df = self.parseRegistry(self.sHivePath)

            print("Applying labels...")
            df = self.applyLabels(df)

            print("Preprocessing data...")
            df = self.preprocessData(df)

            # Raw Parsed CSV
            if self.sRawParsedCsvPath and os.path.exists(self.sRawParsedCsvPath):
                self.appendToExistingCsv(df, self.sRawParsedCsvPath)
                print(f"Appended data to existing raw parsed CSV: {self.sRawParsedCsvPath}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_parsed_csv_path = os.path.join(self.sModelOutputDir, f"raw_parsed_{timestamp}.csv")
                df.to_csv(new_parsed_csv_path, index=False)
                self.sRawParsedCsvPath = new_parsed_csv_path
                self.rawParsedCsvInput.delete(0, tk.END)
                self.rawParsedCsvInput.insert(0, new_parsed_csv_path)
                print(f"New raw parsed CSV created: {new_parsed_csv_path}")

            # Training Dataset
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                self.appendToExistingCsv(df, self.sTrainingDatasetPath)
                print(f"Appended data to existing training dataset: {self.sTrainingDatasetPath}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_training_dataset_path = os.path.join(self.sModelOutputDir, f"training_dataset_{timestamp}.csv")
                df.to_csv(new_training_dataset_path, index=False)
                self.sTrainingDatasetPath = new_training_dataset_path
                self.trainingDatasetInput.delete(0, tk.END)
                self.trainingDatasetInput.insert(0, new_training_dataset_path)
                print(f"New training dataset created: {new_training_dataset_path}")
                messagebox.showinfo("Dataset Created", f"New training dataset created: {new_training_dataset_path}")

        except Exception as e:
            print(f"Error in makeDataset: {e}")
            messagebox.showerror("Error", f"Failed to make dataset: {e}")

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
    #                         APPLY LABELS
    ###########################################################################
    def applyLabels(self, df):
        try:
            if 'Key' not in df.columns:
                raise KeyError("The 'Key' column is missing from the DataFrame.")

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
                row_key = re.sub(r'\\+', r'\\', str(row.get('Key', '') or '').strip().lower())
                row_name = str(row.get('Name', '') or '').strip().lower()
                row_value = re.sub(r'\\+', r'\\', str(row.get('Value', '') or '').strip().lower())
                row_type = str(row.get('Type', '') or '').strip().lower()

                for entry in malicious_entries:
                    entry_key_last = entry['Key'].strip().split('\\')[-1].lower()
                    row_key_last = row_key.split('\\')[-1]

                    if row_key_last != entry_key_last:
                        continue
                    if entry['Name'] and row_name != entry['Name'].lower():
                        continue
                    if entry['Value'] and row_value != entry['Value'].lower():
                        continue
                    if entry['Type'] and row_type != entry['Type'].lower():
                        continue
                    return 'Malicious'
                return 'Benign'

            def assign_tactic(row):
                row_key = re.sub(r'\\+', r'\\', str(row.get('Key', '') or '').strip().lower())
                row_name = str(row.get('Name', '') or '').strip().lower()
                row_value = re.sub(r'\\+', r'\\', str(row.get('Value', '') or '').strip().lower())
                row_type = str(row.get('Type', '') or '').strip().lower()

                for entry in tagged_entries:
                    entry_key_last = entry['Key'].strip().split('\\')[-1].lower()
                    row_key_last = row_key.split('\\')[-1]

                    if row_key_last != entry_key_last:
                        continue
                    if entry['Name'] and row_name != entry['Name'].lower():
                        continue
                    if entry['Value'] and row_value != entry['Value'].lower():
                        continue
                    if entry['Type'] and row_type != entry['Type'].lower():
                        continue
                    return entry['Tactic']
                return 'None'

            df['Label'] = df.apply(is_malicious, axis=1)
            df['Tactic'] = df.apply(assign_tactic, axis=1)
            return df

        except Exception as e:
            err_msg = f"Error applying labels: {e}"
            print(err_msg)
            raise RuntimeError(err_msg)

    ###########################################################################
    #                          PREPROCESS DATA
    ###########################################################################
    def preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            print("No valid data to preprocess.")
            return pd.DataFrame()

        try:
            xDf = df.copy()
            xDf.fillna(xDf.select_dtypes(include=[np.number]).mean(), inplace=True)

            # Encode path category
            xDf['Path Category'] = xDf['Key'].apply(self.categorizePath)
            path_encoded = pd.get_dummies(xDf['Path Category'], prefix='PathCategory')
            xDf = pd.concat([xDf.drop('Path Category', axis=1), path_encoded], axis=1)

            # Encode type group
            xDf['Type Group'] = xDf['Type'].apply(self.mapType)
            type_group_encoded = pd.get_dummies(xDf['Type Group'], prefix='TypeGroup')
            xDf = pd.concat([xDf.drop(['Type', 'Type Group'], axis=1), type_group_encoded], axis=1)

            # Encode key name category
            xDf['Key Name Category'] = xDf['Name'].apply(self.categorizeKeyName)
            key_name_encoded = pd.get_dummies(xDf['Key Name Category'], prefix='KeyNameCategory')
            xDf = pd.concat([xDf.drop('Key Name Category', axis=1), key_name_encoded], axis=1)

            # Convert value
            xDf['Value Processed'] = xDf['Value'].apply(self.preprocessValue)
            xDf.drop('Value', axis=1, inplace=True)

            # Drop Key & Name columns
            xDf.drop(columns=['Key', 'Name'], inplace=True, errors='ignore')

            # Scale numeric
            scaler_minmax = MinMaxScaler()
            minmax_cols = ['Depth', 'Value Count', 'Value Processed']
            xDf[minmax_cols] = scaler_minmax.fit_transform(xDf[minmax_cols])

            scaler_robust = RobustScaler()
            robust_cols = ['Key Size', 'Subkey Count']
            xDf[robust_cols] = scaler_robust.fit_transform(xDf[robust_cols])

            return xDf
        except Exception as e:
            raise RuntimeError(f"Preprocessing error: {e}")

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
    #                          EXECUTE ML PROCESS
    ###########################################################################
    def executeMLProcess(self):
        """
        Main pipeline: if no CSV is provided, we classify the training dataset
        for demonstration. Then we do train->evaluate->classify.
        """
        try:
            # If no classify CSV specified, default to training dataset
            if not self.sClassifyCsvPath:
                print("No CSV to classify provided, using training dataset for testing.")
                self.sClassifyCsvPath = self.sTrainingDatasetPath

            print("Loading training dataset...")
            df_train = pd.read_csv(self.sTrainingDatasetPath)

            self.trainAndEvaluateModels(df_train)
            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")

        except Exception as e:
            messagebox.showerror("Error", f"Error in ML process: {e}")

    ###########################################################################
    #                          TRAIN & EVALUATE
    ###########################################################################
    def trainAndEvaluateModels(self, df):
        """
        Train/evaluate three RF models (Label, Defense, Persistence).
        Make ~30% of the data to another class for testing purposes.
        """
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        try:
            if df.empty:
                raise ValueError("DataFrame is empty.")

            X = df.drop(columns=['Label', 'Tactic'], errors='ignore')
            if 'Label' not in df.columns:
                raise ValueError("Missing 'Label' column in dataset.")
            if 'Tactic' not in df.columns:
                raise ValueError("Missing 'Tactic' column in dataset.")

            y_label = (df['Label'] == 'Malicious').astype(int)
            y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
            y_persistence = (df['Tactic'] == 'Persistence').astype(int)

            # Force multi-class if single-class
            if y_label.nunique() < 2:
                n_lbl = int(len(y_label) * 0.3)
                flip_idx = np.random.choice(y_label.index, size=n_lbl, replace=False)
                y_label.iloc[flip_idx] = 1
                print("Forcing 30% to 'Malicious' in y_label for demonstration.")

            if y_defense.nunique() < 2:
                n_def = int(len(y_defense) * 0.3)
                flip_idx = np.random.choice(y_defense.index, size=n_def, replace=False)
                y_defense.iloc[flip_idx] = 1
                print("Forcing 30% to 'Defense Evasion' in y_defense for demonstration.")

            if y_persistence.nunique() < 2:
                n_per = int(len(y_persistence) * 0.3)
                flip_idx = np.random.choice(y_persistence.index, size=n_per, replace=False)
                y_persistence.iloc[flip_idx] = 1
                print("Forcing 30% to 'Persistence' in y_persistence for demonstration.")

            print("Performing RFE for feature selection...")
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=base_model, n_features_to_select=10)
            rfe.fit(X, y_label)
            self.selected_features = X.columns[rfe.support_]
            print("Selected features:", list(self.selected_features))

            X_selected = X[self.selected_features]

            def grid_search_rf(X_part, y_part):
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False]
                }
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                model = RandomForestClassifier(random_state=42)
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                gs.fit(X_part, y_part)
                return gs.best_estimator_

            def evaluate_model(model, X_data, y_data, label_name=""):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
                )
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
                if y_scores is not None and len(np.unique(y_test)) > 1:
                    aucv = roc_auc_score(y_test, y_scores)

                print(f"{label_name} => Acc: {acc:.4f}, Prec: {prec:.4f}, "
                      f"Rec: {rec:.4f}, F1: {f1v:.4f}, AUC: {aucv:.4f}")

                return {
                    f"{label_name} Accuracy": acc,
                    f"{label_name} Precision": prec,
                    f"{label_name} Recall": rec,
                    f"{label_name} F1": f1v,
                    f"{label_name} AUC": aucv
                }

            # Label Model
            print("Training Label Model...")
            label_model = grid_search_rf(X_selected, y_label)
            label_metrics = evaluate_model(label_model, X_selected, y_label, "Label Model")
            label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
            joblib.dump(label_model, label_model_path)
            self.sLabelModelPath = label_model_path

            # Defense Evasion Model
            print("Training Defense Evasion Model...")
            defense_model = grid_search_rf(X_selected, y_defense)
            defense_metrics = evaluate_model(defense_model, X_selected, y_defense, "Defense Evasion Model")
            defense_model_path = os.path.join(self.sModelOutputDir, "defense_model.joblib")
            joblib.dump(defense_model, defense_model_path)
            self.sTacticModelPath = defense_model_path

            # Persistence Model
            print("Training Persistence Model...")
            persistence_model = grid_search_rf(X_selected, y_persistence)
            persist_metrics = evaluate_model(persistence_model, X_selected, y_persistence, "Persistence Model")
            persistence_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
            joblib.dump(persistence_model, persistence_model_path)
            self.sPersistenceModelPath = persistence_model_path

            # Combine metrics
            all_metrics = {}
            all_metrics.update(label_metrics)
            all_metrics.update(defense_metrics)
            all_metrics.update(persist_metrics)

            # Convert to string for UI
            string_metrics = {k: f"{v:.4f}" for k, v in all_metrics.items()}
            self.updateMetricsDisplay(string_metrics)

        except Exception as e:
            raise RuntimeError(f"Training error: {e}")

    ###########################################################################
    #                              CLASSIFY CSV
    ###########################################################################
    def classifyCsv(self, csv_path):
        """
        Classifies a new CSV using the three trained models.
        Uses explicit check for 'selected_features' to avoid ambiguous Index errors.
        """
        try:
            df = pd.read_csv(csv_path)

            # Ensure self.selected_features is valid
            if self.selected_features is None or len(self.selected_features) == 0:
                raise ValueError("No selected_features found. Did you run the training process?")

            # Load the 3 models
            label_model = joblib.load(self.sLabelModelPath)
            defense_model = joblib.load(self.sTacticModelPath)
            persistence_model = joblib.load(self.sPersistenceModelPath)

            # Filter features
            X = df[self.selected_features].copy()

            # Label classification
            y_scores_label = label_model.predict_proba(X)[:, 1]
            y_pred_label = np.where(y_scores_label >= 0.5, 'Malicious', 'Benign')

            # Defense Evasion
            y_scores_defense = defense_model.predict_proba(X)[:, 1]
            y_pred_defense = np.where(y_scores_defense >= 0.5, 'Defense Evasion', 'None')

            # Persistence
            y_scores_persistence = persistence_model.predict_proba(X)[:, 1]
            y_pred_persistence = np.where(y_scores_persistence >= 0.5, 'Persistence', 'None')

            # Combine Tactic
            df['Predicted Label'] = y_pred_label
            df['Predicted Tactic'] = np.where(
                y_pred_defense == 'Defense Evasion',
                'Defense Evasion',
                y_pred_persistence
            )

            # Save classified output
            output_path = os.path.join(
                self.sModelOutputDir,
                f"classified_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            df.to_csv(output_path, index=False)
            print(f"Classified output saved to: {output_path}")
            messagebox.showinfo("Classification Complete", f"Classified output saved to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"Classification error: {e}")

    ###########################################################################
    #                           METRICS DISPLAY
    ###########################################################################
    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, value in metrics.items():
            try:
                val_str = f"{float(value):.4f}"
            except (ValueError, TypeError):
                val_str = str(value)
            self.metricsList.insert("", "end", values=(metric, val_str))


if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
