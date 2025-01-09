import os
import tkinter as tk
import pandas as pd
import numpy as np
import joblib

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
        """
        Parse the registry, apply labels, preprocess the data, and save or append the dataset.
        Ensures the same CSV is used for both preprocessed and training data.
        """
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
        try:
            xData = []
            subkey_counts = {}
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
            raise RuntimeError(f"Failed to parse registry: {e}")

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

            if os.path.exists(self.sMaliciousKeysPath):
                with open(self.sMaliciousKeysPath, 'r') as f:
                    malicious_keys = set(f.read().splitlines())
            else:
                malicious_keys = set()

            if os.path.exists(self.sTaggedKeysPath):
                with open(self.sTaggedKeysPath, 'r') as f:
                    tagged_keys = {line.split('[')[0].strip(): line.split('[')[1].strip(']') for line in f if '[' in line}
            else:
                tagged_keys = {}

            if not malicious_keys:
                print("No malicious keys provided. Randomly labeling entries.")
                df['Label'] = np.random.choice(['Benign', 'Malicious'], size=len(df))
            else:
                df['Label'] = df['Key'].apply(lambda x: 'Malicious' if x in malicious_keys else 'Benign')

            if not tagged_keys:
                print("No tagged keys provided. Assigning random tactics.")
                possible_tactics = ['Persistence', 'Execution', 'Privilege Escalation', 'Defense Evasion']
                df['Tactic'] = np.random.choice(possible_tactics, size=len(df))
            else:
                df['Tactic'] = df.apply(lambda row: tagged_keys.get(row['Key'], 'Persistence') if row['Label'] == 'Malicious' else 'None', axis=1)

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

            # Recursive Feature Elimination (RFE) for feature selection
            print("Performing RFE for feature selection...")
            rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=rfe_model, n_features_to_select=10)
            X_selected = rfe.fit_transform(X, y_label)
            self.selected_features = X.columns[rfe.support_]
            print(f"Selected features: {list(self.selected_features)}")

            # Convert X_selected back to DataFrame with selected feature names
            X_selected_df = pd.DataFrame(X_selected, columns=self.selected_features)

            skf = StratifiedKFold(n_splits=10)

            # Train and evaluate Label Model
            label_model = RandomForestClassifier(n_estimators=100, random_state=42)
            label_scores = cross_val_score(label_model, X_selected_df, y_label, cv=skf)
            label_model.fit(X_selected_df, y_label)
            joblib.dump(label_model, os.path.join(self.sModelOutputDir, "label_model.joblib"))
            self.sLabelModelPath = os.path.join(self.sModelOutputDir, "label_model.joblib")

            # Calculate additional metrics for Label Model
            y_pred_label = label_model.predict(X_selected_df)
            precision_label = precision_score(y_label, y_pred_label)
            recall_label = recall_score(y_label, y_pred_label)
            f1_label = f1_score(y_label, y_pred_label)
            
            # ROC AUC and optimal threshold for Label Model
            y_scores_label = label_model.predict_proba(X_selected_df)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_label, y_scores_label)
            auc_val = roc_auc_score(y_label, y_scores_label)
            opt_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
            self.opt_threshold_label = thresholds[opt_idx]
            
            print(f"Label Model Metrics:\n  K-Fold Accuracy: {label_scores.mean():.4f}\n  Precision: {precision_label:.4f}\n  Recall: {recall_label:.4f}\n  F1 Score: {f1_label:.4f}\n  ROC AUC: {auc_val:.4f}\n  Optimal Threshold: {self.opt_threshold_label:.4f}")

            # Train and evaluate Defense Evasion Model
            defense_model = RandomForestClassifier(n_estimators=100, random_state=42)
            defense_scores = cross_val_score(defense_model, X_selected_df, y_defense, cv=skf)
            defense_model.fit(X_selected_df, y_defense)
            joblib.dump(defense_model, os.path.join(self.sModelOutputDir, "defense_evasion_model.joblib"))
            self.sTacticModelPath = os.path.join(self.sModelOutputDir, "defense_evasion_model.joblib")

            # Calculate additional metrics for Defense Evasion Model
            y_pred_defense = defense_model.predict(X_selected_df)
            precision_defense = precision_score(y_defense, y_pred_defense)
            recall_defense = recall_score(y_defense, y_pred_defense)
            f1_defense = f1_score(y_defense, y_pred_defense)

            # ROC AUC and optimal threshold for Defense Evasion Model
            y_scores_defense = defense_model.predict_proba(X_selected_df)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_defense, y_scores_defense)
            auc_val_defense = roc_auc_score(y_defense, y_scores_defense)
            opt_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
            self.opt_threshold_defense = thresholds[opt_idx]
            
            print(f"Defense Evasion Model Metrics:\n  K-Fold Accuracy: {defense_scores.mean():.4f}\n  Precision: {precision_defense:.4f}\n  Recall: {recall_defense:.4f}\n  F1 Score: {f1_defense:.4f}\n  ROC AUC: {auc_val_defense:.4f}\n  Optimal Threshold: {self.opt_threshold_defense:.4f}")

            # Train and evaluate Persistence Model
            persistence_model = RandomForestClassifier(n_estimators=100, random_state=42)
            persistence_scores = cross_val_score(persistence_model, X_selected_df, y_persistence, cv=skf)
            persistence_model.fit(X_selected_df, y_persistence)
            joblib.dump(persistence_model, os.path.join(self.sModelOutputDir, "persistence_model.joblib"))
            self.sPersistenceModelPath = os.path.join(self.sModelOutputDir, "persistence_model.joblib")

            # Calculate additional metrics for Persistence Model
            y_pred_persistence = persistence_model.predict(X_selected_df)
            precision_persistence = precision_score(y_persistence, y_pred_persistence)
            recall_persistence = recall_score(y_persistence, y_pred_persistence)
            f1_persistence = f1_score(y_persistence, y_pred_persistence)

            # ROC AUC and optimal threshold for Persistence Model
            y_scores_persistence = persistence_model.predict_proba(X_selected_df)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_persistence, y_scores_persistence)
            auc_val_persistence = roc_auc_score(y_persistence, y_scores_persistence)
            opt_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
            self.opt_threshold_persistence = thresholds[opt_idx]
            
            print(f"Persistence Model Metrics:\n  K-Fold Accuracy: {persistence_scores.mean():.4f}\n  Precision: {precision_persistence:.4f}\n  Recall: {recall_persistence:.4f}\n  F1 Score: {f1_persistence:.4f}\n  ROC AUC: {auc_val_persistence:.4f}\n  Optimal Threshold: {self.opt_threshold_persistence:.4f}")

            # Display metrics in GUI
            metrics = {
                "Label Model K-Fold Accuracy": f"{label_scores.mean():.4f}",
                "Label Model Precision": f"{precision_label:.4f}",
                "Label Model Recall": f"{recall_label:.4f}",
                "Label Model F1 Score": f"{f1_label:.4f}",
                "Label Model ROC AUC": f"{auc_val:.4f}",
                "Defense Evasion Model K-Fold Accuracy": f"{defense_scores.mean():.4f}",
                "Defense Evasion Model Precision": f"{precision_defense:.4f}",
                "Defense Evasion Model Recall": f"{recall_defense:.4f}",
                "Defense Evasion Model F1 Score": f"{f1_defense:.4f}",
                "Defense Evasion Model ROC AUC": f"{auc_val_defense:.4f}",
                "Persistence Model K-Fold Accuracy": f"{persistence_scores.mean():.4f}",
                "Persistence Model Precision": f"{precision_persistence:.4f}",
                "Persistence Model Recall": f"{recall_persistence:.4f}",
                "Persistence Model F1 Score": f"{f1_persistence:.4f}",
                "Persistence Model ROC AUC": f"{auc_val_persistence:.4f}"
            }
            self.updateMetricsDisplay(metrics)

        except Exception as e:
            raise RuntimeError(f"Training error: {e}")

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
        try:
            # Clear the existing metrics list
            for item in self.metricsList.get_children():
                self.metricsList.delete(item)

            # Insert the new metrics into the list
            for metric, value in metrics.items():
                self.metricsList.insert("", "end", values=(metric, value))

            print("Metrics display updated successfully.")
        except Exception as e:
            print(f"Error updating metrics display: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()
