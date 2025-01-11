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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, roc_auc_score
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
        self.sLabelModelPath = ''
        self.sTacticModelPath = ''
        self.sPersistenceModelPath = ''

        self.model_loaded = False

        self.setupUI()

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

        ttk.Label(frame, text="CSV to Classify (Optional):").grid(row=4, column=0, sticky='e')
        self.classifyCsvInput = ttk.Entry(frame, width=50)
        self.classifyCsvInput.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Set CSV to Classify", command=self.setClassifyCsvPath).grid(row=4, column=2, padx=5)

        ttk.Label(frame, text="Label Model (Optional):").grid(row=5, column=0, sticky='e')
        self.labelModelInput = ttk.Entry(frame, width=50)
        self.labelModelInput.grid(row=5, column=1, padx=5)
        ttk.Button(frame, text="Set Label Model", command=self.setLabelModelPath).grid(row=5, column=2, padx=5)

        ttk.Label(frame, text="Defense Evasion Model (Optional):").grid(row=6, column=0, sticky='e')
        self.tacticModelInput = ttk.Entry(frame, width=50)
        self.tacticModelInput.grid(row=6, column=1, padx=5)
        ttk.Button(frame, text="Set Defense Evasion Model", command=self.setTacticModelPath).grid(row=6, column=2, padx=5)

        ttk.Label(frame, text="Persistence Model (Optional):").grid(row=7, column=0, sticky='e')
        self.persistenceModelInput = ttk.Entry(frame, width=50)
        self.persistenceModelInput.grid(row=7, column=1, padx=5)
        ttk.Button(frame, text="Set Persistence Model", command=self.setPersistenceModelPath).grid(row=7, column=2, padx=5)

        ttk.Button(frame, text="Make Dataset", command=self.makeDataset).grid(row=8, column=1, pady=10)
        ttk.Button(frame, text="Start ML Process", command=self.executeMLProcess).grid(row=9, column=1, pady=10)

        self.metricsList = ttk.Treeview(frame, columns=("Metric", "Value"), show="headings")
        self.metricsList.heading("Metric", text="Metric")
        self.metricsList.heading("Value", text="Value")
        self.metricsList.column("Metric", width=200, anchor="w")
        self.metricsList.column("Value", width=500, anchor="w")
        self.metricsList.grid(row=10, column=0, columnspan=3, pady=10)

    def setPersistenceModelPath(self):
        self.sPersistenceModelPath = self.persistenceModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Persistence model set to: {self.sPersistenceModelPath}")

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

    def setClassifyCsvPath(self):
        self.sClassifyCsvPath = self.classifyCsvInput.get().strip()
        messagebox.showinfo("Path Set", f"CSV to classify set to: {self.sClassifyCsvPath}")

    def setLabelModelPath(self):
        self.sLabelModelPath = self.labelModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Label model set to: {self.sLabelModelPath}")

    def setTacticModelPath(self):
        self.sTacticModelPath = self.tacticModelInput.get().strip()
        messagebox.showinfo("Path Set", f"Tactic model set to: {self.sTacticModelPath}")

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

            # Check if training dataset path exists, append if valid
            if self.sTrainingDatasetPath and os.path.exists(self.sTrainingDatasetPath):
                existing_df = pd.read_csv(self.sTrainingDatasetPath)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(self.sTrainingDatasetPath, index=False)
                print(f"Appended preprocessed data to existing training dataset: {self.sTrainingDatasetPath}")
            else:
                # Create new training dataset if none exists
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_dataset_path = os.path.join(self.sModelOutputDir, f"training_dataset_{timestamp}.csv")
                df.to_csv(new_dataset_path, index=False)
                self.sTrainingDatasetPath = new_dataset_path
                self.trainingDatasetInput.delete(0, tk.END)
                self.trainingDatasetInput.insert(0, new_dataset_path)
                print(f"New training dataset created: {new_dataset_path}")
                messagebox.showinfo("Dataset Created", f"New training dataset created: {new_dataset_path}")

        except Exception as e:
            print(f"Error in makeDataset: {e}")
            messagebox.showerror("Error", f"Failed to make dataset: {e}")

    def executeMLProcess(self):
        try:
            if not self.sClassifyCsvPath:
                print("No CSV to classify provided. Generating one using the training dataset...")
                df = pd.read_csv(self.sTrainingDatasetPath)
                self.sClassifyCsvPath = os.path.join(self.sModelOutputDir, "generated_classify.csv")
                df.drop(columns=['Label', 'Tactic'], errors='ignore').to_csv(self.sClassifyCsvPath, index=False)

            print("Loading training dataset...")
            df = pd.read_csv(self.sTrainingDatasetPath)

            if not self.sLabelModelPath or not os.path.exists(self.sLabelModelPath):
                print("No existing Label Model found. Training a new Label Model...")
            if not self.sTacticModelPath or not os.path.exists(self.sTacticModelPath):
                print("No existing Tactic Model found. Training a new Tactic Model...")

            self.trainAndEvaluateModels(df)
            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")
        except Exception as e:
            messagebox.showerror("Error", f"Error in ML process: {e}")

    def parseRegistry(self, sHivePath):
        xData = []
        subkey_counts = {}

        try:
            with ThreadPoolExecutor() as executor:
                xHive = RegistryHive(sHivePath)

                for xSubkey in xHive.recurse_subkeys():
                    sKeyPath = xSubkey.path
                    parent_path = '\\'.join(sKeyPath.split('\\')[:-1])
                    if parent_path in subkey_counts:
                        subkey_counts[parent_path] += 1
                    else:
                        subkey_counts[parent_path] = 1

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
            
            # Export the parsed data
            self.exportToCSV(xData, 'raw')
            
            # Preprocess and labelling of the data will utilize the returned dataframe
            return pd.DataFrame(xData)

        except Exception as e:
            messagebox.showerror("Error", f"Error parsing hive: {e}")
        return xData

    def exportToCSV(self, xData, sPrefix):
        columns = ["Key", "Name", "Value", "Type", "Subkey Count", "Value Count", "Key Size", "Depth"]
        xDf = pd.DataFrame(xData, columns=columns)

        xDf.dropna(axis=1, how='all', inplace=True)

        sTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sOutputCsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{sPrefix}_{sTimestamp}.csv")
        
        xDf.to_csv(sOutputCsv, index=False)
        messagebox.showinfo("Export Complete", f"Data exported to: {sOutputCsv}")

    def appendToExistingCsv(self, new_df: pd.DataFrame, sAppendPath: str) -> None:
        try:
            if not sAppendPath or not os.path.exists(sAppendPath):
                new_df.to_csv(sAppendPath, index=False)
                print(f"Data saved to new CSV: {sAppendPath}")
                return

            existing_df = pd.read_csv(sAppendPath)
            if set(existing_df.columns) != set(new_df.columns):
                print("Columns mismatch. Saving to new CSV.")
                new_df.to_csv(sAppendPath, index=False)
                return

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(sAppendPath, index=False)
            print(f"Data appended to existing CSV: {sAppendPath}")

        except Exception as e:
            print(f"Error appending to CSV ({sAppendPath}): {e}")
            new_df.to_csv(sAppendPath, index=False)

    def preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            print("No valid data to preprocess.")
            return pd.DataFrame()

        try:
            xDf = df.copy()
            xDf.fillna(xDf.select_dtypes(include=[np.number]).mean(), inplace=True)

            # Encode Path Category
            xDf['Path Category'] = xDf['Key'].apply(self.categorizePath)
            path_encoded = pd.get_dummies(xDf['Path Category'], prefix='PathCategory')
            xDf = pd.concat([xDf.drop('Path Category', axis=1), path_encoded], axis=1)

            # Encode Type Group
            xDf['Type Group'] = xDf['Type'].apply(self.mapType)
            type_group_encoded = pd.get_dummies(xDf['Type Group'], prefix='TypeGroup')
            xDf = pd.concat([xDf.drop(['Type', 'Type Group'], axis=1), type_group_encoded], axis=1)

            # Encode Key Name Category
            xDf['Key Name Category'] = xDf['Name'].apply(self.categorizeKeyName)
            key_name_encoded = pd.get_dummies(xDf['Key Name Category'], prefix='KeyNameCategory')
            xDf = pd.concat([xDf.drop('Key Name Category', axis=1), key_name_encoded], axis=1)

            # Convert Value to numeric
            xDf['Value Processed'] = xDf['Value'].apply(self.preprocessValue)
            xDf.drop('Value', axis=1, inplace=True)

            # Drop Key and Name columns (no longer needed for ML)
            xDf.drop(columns=['Key', 'Name'], inplace=True, errors='ignore')

            # Scale numeric features
            scaler_minmax = MinMaxScaler()
            minmax_cols = ['Depth', 'Value Count', 'Value Processed']
            xDf[minmax_cols] = scaler_minmax.fit_transform(xDf[minmax_cols])

            scaler_robust = RobustScaler()
            robust_cols = ['Key Size', 'Subkey Count']
            xDf[robust_cols] = scaler_robust.fit_transform(xDf[robust_cols])

            return xDf

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise RuntimeError(f"Preprocessing error: {e}")

    def categorizePath(self, path):
        if "Run" in path:
            return "Startup Path"
        elif "Services" in path:
            return "Service Path"
        elif "Internet Settings" in path:
            return "Network Path"
        else:
            return "Other Path"

    def mapType(self, value_type):
        type_grouping = {
            "String": ["REG_SZ", "REG_EXPAND_SZ", "REG_MULTI_SZ"],
            "Numeric": ["REG_DWORD", "REG_QWORD"],
            "Binary": ["REG_BINARY"],
            "Others": ["REG_NONE", "REG_LINK"]
        }
        for group, reg_types in type_grouping.items():
            if value_type in reg_types:
                return group
        return "Others"

    def preprocessValue(self, value):
        if isinstance(value, str):
            return len(value)
        elif isinstance(value, (int, float)):
            return value
        else:
            return 0

    def categorizeKeyName(self, key_name):
        key_name_categories = {
            "Run Keys": ["Run", "RunOnce", "RunServices"],
            "Service Keys": ["ImageFileExecutionOptions", "AppInit_DLLs"],
            "Security and Configuration Keys": ["Policies", "Explorer"],
            "Internet and Network Keys": ["ProxyEnable", "ProxyServer"],
            "File Execution Keys": ["ShellExecuteHooks"]
        }
        for category, keywords in key_name_categories.items():
            if any(keyword in key_name for keyword in keywords):
                return category
        return "Other Keys"
            
    def applyLabels(self, df):
        try:
            if 'Key' not in df.columns:
                raise KeyError("The 'Key' column is missing from the DataFrame.")

            # Load malicious entries with attributes
            malicious_entries = []
            if os.path.exists(self.sMaliciousKeysPath):
                with open(self.sMaliciousKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[,\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": parts[0].replace('\\\\', '\\'),  # Replace double backslashes in Key
                            "Name": parts[1] if len(parts) > 1 and parts[1].lower() != "none" else None,
                            "Value": parts[2].replace('\\\\', '\\') if len(parts) > 2 and parts[2].lower() != "none" else None,
                            "Type": parts[3] if len(parts) > 3 and parts[3].lower() != "none" else None
                        }
                        malicious_entries.append(entry)

            print("Loaded malicious entries:", malicious_entries)  # Debug log

            # Load tagged entries with attributes and tactics
            tagged_entries = []
            if os.path.exists(self.sTaggedKeysPath):
                with open(self.sTaggedKeysPath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in re.split(r'[,\|;]', line.strip()) if p.strip()]
                        entry = {
                            "Key": parts[0].replace('\\\\', '\\'),  # Replace double backslashes in Key
                            "Name": parts[1] if len(parts) > 1 and parts[1].lower() != "none" else None,
                            "Value": parts[2].replace('\\\\', '\\') if len(parts) > 2 and parts[2].lower() != "none" else None,
                            "Type": parts[3] if len(parts) > 3 and parts[3].lower() != "none" else None,
                            "Tactic": parts[4] if len(parts) > 4 else "Persistence"
                        }
                        tagged_entries.append(entry)

            print("Loaded tagged entries:", tagged_entries)  # Debug log

            # Function to check if a row matches any malicious entry
            def is_malicious(row):
                row_key = row['Key']
                row_name = row.get('Name', '')
                row_value = str(row.get('Value', ''))
                row_type = row.get('Type', '')

                for entry in malicious_entries:
                    # Check only non-None attributes
                    if entry['Key'] and row_key != entry['Key']:
                        continue
                    if entry['Name'] and row_name != entry['Name']:
                        continue
                    if entry['Value'] and row_value != entry['Value']:
                        continue
                    if entry['Type'] and row_type != entry['Type']:
                        continue
                    print(f"Matched malicious entry: {row_key}")  # Debug log
                    return 'Malicious'
                return 'Benign'

            # Function to assign a tactic based on any tagged entry
            def assign_tactic(row):
                row_key = row['Key']
                row_name = row.get('Name', '')
                row_value = str(row.get('Value', ''))
                row_type = row.get('Type', '')

                for entry in tagged_entries:
                    # Check only non-None attributes
                    if entry['Key'] and row_key != entry['Key']:
                        continue
                    if entry['Name'] and row_name != entry['Name']:
                        continue
                    if entry['Value'] and row_value != entry['Value']:
                        continue
                    if entry['Type'] and row_type != entry['Type']:
                        continue
                    print(f"Matched tagged entry: {row_key} with tactic: {entry['Tactic']}")  # Debug log
                    return entry['Tactic']
                return 'None'

            # Apply the labeling and tactic assignment functions
            df['Label'] = df.apply(is_malicious, axis=1)
            df['Tactic'] = df.apply(assign_tactic, axis=1)

            return df

        except Exception as e:
            print(f"Error applying labels: {e}")
            raise

    def executeMLProcess(self):
        try:
            if not self.sClassifyCsvPath:
                print("No CSV to classify provided. Generating one using the training dataset...")
                df = pd.read_csv(self.sTrainingDatasetPath)
                self.sClassifyCsvPath = os.path.join(self.sModelOutputDir, "generated_classify.csv")
                df.drop(columns=['Label', 'Tactic'], errors='ignore').to_csv(self.sClassifyCsvPath, index=False)

            print("Loading training dataset...")
            df = pd.read_csv(self.sTrainingDatasetPath)

            if not self.sLabelModelPath or not os.path.exists(self.sLabelModelPath):
                print("No existing Label Model found. Training a new Label Model...")
            if not self.sTacticModelPath or not os.path.exists(self.sTacticModelPath):
                print("No existing Defense Evasion Model found. Training a new Defense Evasion Model...")
            if not self.sPersistenceModelPath or not os.path.exists(self.sPersistenceModelPath):
                print("No existing Persistence Model found. Training a new Persistence Model...")

            self.trainAndEvaluateModels(df)
            print("Classifying the provided CSV...")
            self.classifyCsv(self.sClassifyCsvPath)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")
        except Exception as e:
            messagebox.showerror("Error", f"Error in ML process: {e}")

    def trainAndEvaluateModels(self, df):
        try:
            if df.empty:
                raise ValueError("DataFrame is empty")

            # Prepare features and labels
            X = df.drop(columns=['Label', 'Tactic'])
            y_label = (df['Label'] == 'Malicious').astype(int)
            y_defense = (df['Tactic'] == 'Defense Evasion').astype(int)
            y_persistence = (df['Tactic'] == 'Persistence').astype(int)

            # Perform Recursive Feature Elimination (RFE) for feature selection
            print("Performing RFE for feature selection...")
            rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=rfe_model, n_features_to_select=10)
            X_selected = rfe.fit_transform(X, y_label)
            self.selected_features = df.drop(columns=['Label', 'Tactic']).columns[rfe.support_]
            print(f"Selected features: {list(self.selected_features)}")

            # Convert X_selected back to DataFrame with selected feature names
            X_selected_df = pd.DataFrame(X_selected, columns=self.selected_features)

            # Split the data into training, validation, and test sets
            X_train, X_temp, y_train_label, y_temp_label = train_test_split(X_selected_df, y_label, test_size=0.3, random_state=42)
            X_val, X_test, y_val_label, y_test_label = train_test_split(X_temp, y_temp_label, test_size=0.5, random_state=42)

            skf = StratifiedKFold(n_splits=10)

            # Train and evaluate Label Model using K-Fold cross-validation
            label_model = RandomForestClassifier(n_estimators=100, random_state=42)
            label_scores = cross_val_score(label_model, X_train, y_train_label, cv=skf)
            label_model.fit(X_train, y_train_label)

            # Evaluate on validation set and compute optimal threshold for Label Model
            y_val_scores_label = label_model.predict_proba(X_val)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_val_label, y_val_scores_label)
            auc_val_label = roc_auc_score(y_val_label, y_val_scores_label)
            opt_idx_label = np.argmax(tpr - fpr)  # Youden's J statistic for optimal threshold
            self.opt_threshold_label = thresholds[opt_idx_label]
            print(f"Label Model ROC AUC: {auc_val_label:.4f}, Optimal Threshold: {self.opt_threshold_label:.4f}")

            # Save Label Model
            label_model_path = os.path.join(self.sModelOutputDir, "label_model.joblib")
            joblib.dump(label_model, label_model_path)
            self.sLabelModelPath = label_model_path
            self.labelModelInput.delete(0, tk.END)
            self.labelModelInput.insert(0, label_model_path)

            # Train and evaluate Defense Evasion Model
            defense_model = RandomForestClassifier(n_estimators=100, random_state=42)
            defense_scores = cross_val_score(defense_model, X_train, y_defense.iloc[y_train_label.index], cv=skf)
            defense_model.fit(X_train, y_defense.iloc[y_train_label.index])

            # Evaluate on validation set and compute optimal threshold for Defense Evasion Model
            y_val_scores_defense = defense_model.predict_proba(X_val)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_val_label, y_val_scores_defense)
            auc_val_defense = roc_auc_score(y_val_label, y_val_scores_defense)
            opt_idx_defense = np.argmax(tpr - fpr)
            self.opt_threshold_defense = thresholds[opt_idx_defense]
            print(f"Defense Evasion Model ROC AUC: {auc_val_defense:.4f}, Optimal Threshold: {self.opt_threshold_defense:.4f}")

            # Save Defense Evasion Model
            defense_model_path = os.path.join(self.sModelOutputDir, "defense_evasion_model.joblib")
            joblib.dump(defense_model, defense_model_path)
            self.sTacticModelPath = defense_model_path
            self.tacticModelInput.delete(0, tk.END)
            self.tacticModelInput.insert(0, defense_model_path)

            # Train and evaluate Persistence Model
            persistence_model = RandomForestClassifier(n_estimators=100, random_state=42)
            persistence_scores = cross_val_score(persistence_model, X_train, y_persistence.iloc[y_train_label.index], cv=skf)
            persistence_model.fit(X_train, y_persistence.iloc[y_train_label.index])

            # Evaluate on validation set and compute optimal threshold for Persistence Model
            y_val_scores_persistence = persistence_model.predict_proba(X_val)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_val_label, y_val_scores_persistence)
            auc_val_persistence = roc_auc_score(y_val_label, y_val_scores_persistence)
            opt_idx_persistence = np.argmax(tpr - fpr)
            self.opt_threshold_persistence = thresholds[opt_idx_persistence]
            print(f"Persistence Model ROC AUC: {auc_val_persistence:.4f}, Optimal Threshold: {self.opt_threshold_persistence:.4f}")

            # Save Persistence Model
            persistence_model_path = os.path.join(self.sModelOutputDir, "persistence_model.joblib")
            joblib.dump(persistence_model, persistence_model_path)
            self.sPersistenceModelPath = persistence_model_path
            self.persistenceModelInput.delete(0, tk.END)
            self.persistenceModelInput.insert(0, persistence_model_path)

            # Display metrics
            metrics = {
                "Label Model K-Fold Accuracy": f"{float(label_scores.mean()):.4f}",
                "Defense Evasion Model K-Fold Accuracy": f"{float(defense_scores.mean()):.4f}",
                "Persistence Model K-Fold Accuracy": f"{float(persistence_scores.mean()):.4f}",
                "Label Model ROC AUC": f"{auc_val_label:.4f}",
                "Defense Evasion Model ROC AUC": f"{auc_val_defense:.4f}",
                "Persistence Model ROC AUC": f"{auc_val_persistence:.4f}",
                "Optimal Threshold (Label)": f"{self.opt_threshold_label:.4f}",
                "Optimal Threshold (Defense Evasion)": f"{self.opt_threshold_defense:.4f}",
                "Optimal Threshold (Persistence)": f"{self.opt_threshold_persistence:.4f}"
            }
            self.updateMetricsDisplay(metrics)

        except Exception as e:
            raise RuntimeError(f"Training error: {e}")

    def compute_metrics(self, y_true, y_pred, y_scores):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_scores),
            "False Positive Rate": 1 - precision_score(y_true, y_pred),
            "True Positive Rate": recall_score(y_true, y_pred),
        }
        return metrics

    def classifyCsv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            label_model = joblib.load(self.sLabelModelPath)
            defense_model = joblib.load(self.sTacticModelPath)
            persistence_model = joblib.load(self.sPersistenceModelPath)

            X = df[self.selected_features]

            y_scores_label = label_model.predict_proba(X)[:, 1]
            y_pred_label = np.where(y_scores_label >= self.opt_threshold_label, 'Malicious', 'Benign')

            y_scores_defense = defense_model.predict_proba(X)[:, 1]
            y_pred_defense = np.where(y_scores_defense >= self.opt_threshold_defense, 'Defense Evasion', 'None')

            y_scores_persistence = persistence_model.predict_proba(X)[:, 1]
            y_pred_persistence = np.where(y_scores_persistence >= self.opt_threshold_persistence, 'Persistence', 'None')

            df['Predicted Label'] = y_pred_label
            df['Predicted Tactic'] = np.where(y_pred_defense == 'Defense Evasion', 'Defense Evasion', y_pred_persistence)

            output_path = os.path.join(self.sModelOutputDir, f"classified_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(output_path, index=False)
            print(f"Classified output saved to: {output_path}")
            messagebox.showinfo("Classification Complete", f"Classified output saved to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"Classification error: {e}")

    def updateMetricsDisplay(self, metrics):
        self.metricsList.delete(*self.metricsList.get_children())
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    try:
                        formatted_value = f"{float(sub_value):.4f}"
                    except (ValueError, TypeError):
                        formatted_value = str(sub_value)
                    self.metricsList.insert("", "end", values=(f"{metric} - {sub_metric}", formatted_value))
            else:
                try:
                    formatted_value = f"{float(value):.4f}"
                except (ValueError, TypeError):
                    formatted_value = str(value)
                self.metricsList.insert("", "end", values=(metric, formatted_value))

if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
