import os
import subprocess
import tkinter as tk
import pandas as pd
import numpy as np
import re
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

# Set script path references
sScriptPath = os.path.dirname(os.path.abspath(__file__))
sUsername = os.getlogin()
sNtuserPath = rf"C:\Users\{sUsername}\ntuser.dat"  # Example path

class MILOTIC:
    def __init__(self, root):
        self.root = root
        self.root.title("MILOTIC")
        self.root.geometry("1100x750")

        # --- App states & defaults
        self.bAutoRefresh = False
        self.xPreviousData = None
        self.sPreviousHiveType = None
        self.sHivePath = ''
        self.sModelPath = ''  # Store user-specified model path
        self.nEntryLimit = 100
        self.nInterval = 300

        # Two separate paths to optionally append CSV
        self.sAppendParsedCsvPath = ''         # For parsed data
        self.sAppendPreprocessedCsvPath = ''   # For preprocessed data

        # --- Build GUI
        self.setupUI()

    ###########################################################################
    #                              GUI Setup
    ###########################################################################
    def setupUI(self):
        """Build a more compact UI."""
        # ----------------- Input Frame -----------------
        xInputFrame = ttk.Frame(self.root)
        xInputFrame.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        # -- Row 0 --

        # (1) Hive Path
        ttk.Label(xInputFrame, text="Hive Path:").grid(row=0, column=0, padx=2, pady=2, sticky='e')
        self.xHivePathInputBox = ttk.Entry(xInputFrame, width=30)
        self.xHivePathInputBox.grid(row=0, column=1, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Path", command=self.setHivePath).grid(row=0, column=2, padx=2, pady=2, sticky='w')

        # (2) Entry Limit
        ttk.Label(xInputFrame, text="Entry Limit:").grid(row=0, column=3, padx=(10,2), pady=2, sticky='e')
        self.xEntryLimitInput = ttk.Entry(xInputFrame, width=6)
        self.xEntryLimitInput.insert(tk.END, str(self.nEntryLimit))
        self.xEntryLimitInput.grid(row=0, column=4, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Limit", command=self.setEntryLimit).grid(row=0, column=5, padx=2, pady=2, sticky='w')

        # (3) Auto-refresh interval
        ttk.Label(xInputFrame, text="Auto-Refresh (s):").grid(row=0, column=6, padx=(10,2), pady=2, sticky='e')
        self.xIntervalInput = ttk.Entry(xInputFrame, width=6)
        self.xIntervalInput.insert(tk.END, str(self.nInterval))
        self.xIntervalInput.grid(row=0, column=7, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Interval", command=self.setInterval).grid(row=0, column=8, padx=2, pady=2, sticky='w')

        # (4) Start ML Process
        ttk.Button(xInputFrame, text="Start ML Process", command=self.executeMLProcess).grid(
            row=0, column=9, padx=(10,5), pady=2, sticky='w'
        )

        # -- Row 1 -- (Model Path)
        ttk.Label(xInputFrame, text="Model Path:").grid(row=1, column=0, padx=(0,2), pady=2, sticky='e')
        self.xModelPathInputBox = ttk.Entry(xInputFrame, width=30)
        self.xModelPathInputBox.grid(row=1, column=1, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Model Path", command=self.setModelPath).grid(row=1, column=2, padx=2, pady=2, sticky='w')

        # -- Row 2 -- (Append CSV Path: PARSED)
        ttk.Label(xInputFrame, text="Append Parsed CSV (Opt):").grid(row=2, column=0, padx=2, pady=2, sticky='e')
        self.xAppendParsedCsvInputBox = ttk.Entry(xInputFrame, width=30)
        self.xAppendParsedCsvInputBox.grid(row=2, column=1, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Parsed CSV", command=self.setAppendParsedCsvPath).grid(row=2, column=2, padx=2, pady=2, sticky='w')

        # -- Row 3 -- (Append CSV Path: PREPROCESSED)
        ttk.Label(xInputFrame, text="Append Preproc CSV (Opt):").grid(row=3, column=0, padx=2, pady=2, sticky='e')
        self.xAppendPreprocessedCsvInputBox = ttk.Entry(xInputFrame, width=30)
        self.xAppendPreprocessedCsvInputBox.grid(row=3, column=1, padx=2, pady=2, sticky='w')
        ttk.Button(xInputFrame, text="Set Preproc CSV", command=self.setAppendPreprocessedCsvPath).grid(row=3, column=2, padx=2, pady=2, sticky='w')

        # ----------------- Registry Treeview -----------------
        self.xKeyTrees = ttk.Treeview(
            self.root,
            columns=('Name', 'Value', 'Type', 'Subkey Count', 'Value Count', 'Key Size', 'Depth'),
            show='headings',
            selectmode="browse"
        )
        self.xKeyTrees.heading('Name', text='Name', command=lambda: self.sortTreeview('Name', False))
        self.xKeyTrees.heading('Value', text='Value', command=lambda: self.sortTreeview('Value', False))
        self.xKeyTrees.heading('Type', text='Type', command=lambda: self.sortTreeview('Type', False))
        self.xKeyTrees.heading('Subkey Count', text='Subkey Count', command=lambda: self.sortTreeview('Subkey Count', False))
        self.xKeyTrees.heading('Value Count', text='Value Count', command=lambda: self.sortTreeview('Value Count', False))
        self.xKeyTrees.heading('Key Size', text='Key Size', command=lambda: self.sortTreeview('Key Size', False))
        self.xKeyTrees.heading('Depth', text='Depth', command=lambda: self.sortTreeview('Depth', False))

        self.xKeyTrees.column('Name', width=150, anchor='center')
        self.xKeyTrees.column('Value', width=300, anchor='center')
        self.xKeyTrees.column('Type', width=100, anchor='center')
        self.xKeyTrees.column('Subkey Count', width=100, anchor='center')
        self.xKeyTrees.column('Value Count', width=100, anchor='center')
        self.xKeyTrees.column('Key Size', width=100, anchor='center')
        self.xKeyTrees.column('Depth', width=100, anchor='center')

        # Only vertical scrollbar
        xVsb = ttk.Scrollbar(self.root, orient="vertical", command=self.xKeyTrees.yview)
        self.xKeyTrees.configure(yscrollcommand=xVsb.set)

        self.xKeyTrees.grid(row=2, column=0, columnspan=3, sticky='nsew', pady=2)
        xVsb.grid(row=2, column=3, sticky='ns')

        # ----------------- Auto Refresh Button Frame -----------------
        xAutoRefreshButtonFrame = ttk.Frame(self.root)
        xAutoRefreshButtonFrame.grid(row=4, column=0, columnspan=3, pady=5)

        ttk.Button(xAutoRefreshButtonFrame, text="Refresh", command=self.refreshMILOTIC).grid(row=0, column=0, padx=5)
        self.xAutoRefreshButton = ttk.Button(xAutoRefreshButtonFrame, text="Enable Auto Refresh", command=self.toggleAutoRefreshMILOTIC)
        self.xAutoRefreshButton.grid(row=0, column=1, padx=5)

        # ----------------- Changes Frame -----------------
        self.xChangesFrame = ttk.Frame(self.root)
        self.xChangesFrame.grid(row=5, column=0, columnspan=3, sticky='nsew', pady=2)

        self.xChangesList = ttk.Treeview(self.xChangesFrame, columns=('Action', 'Description'), show='headings')
        self.xChangesList.heading('Action', text='Action')
        self.xChangesList.heading('Description', text='Description')
        self.xChangesList.column('Action', width=100, anchor='center')
        self.xChangesList.column('Description', width=800, anchor='w')

        xVsbChanges = ttk.Scrollbar(self.xChangesFrame, orient="vertical", command=self.xChangesList.yview)
        self.xChangesList.configure(yscrollcommand=xVsbChanges.set)

        self.xChangesList.grid(row=0, column=0, sticky='nsew')
        xVsbChanges.grid(row=0, column=1, sticky='ns')

        # ----------------- Metrics Frame -----------------
        self.xMetricsFrame = ttk.Frame(self.root)
        self.xMetricsFrame.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=5, pady=2)

        self.xMetricsList = ttk.Treeview(self.xMetricsFrame, columns=("Metric", "Value"), show="headings")
        self.xMetricsList.heading("Metric", text="Metric")
        self.xMetricsList.heading("Value", text="Value")
        self.xMetricsList.column("Metric", width=200, anchor="w")
        self.xMetricsList.column("Value", width=500, anchor="w")

        self.xMetricsList.grid(row=0, column=0, sticky='nsew')
        xMetricsVsb = ttk.Scrollbar(self.xMetricsFrame, orient="vertical", command=self.xMetricsList.yview)
        self.xMetricsList.configure(yscrollcommand=xMetricsVsb.set)
        xMetricsVsb.grid(row=0, column=1, sticky='ns')

        # Loading Label
        self.xLoadingLabel = ttk.Label(self.root, text="", anchor='center', font=('Arial', 10, 'italic'))
        self.xLoadingLabel.grid(row=7, column=0, columnspan=3, pady=5, sticky='s')

        # --------------- Configure resizing behavior ---------------
        # main tree (parsed data) row => bigger
        self.root.grid_rowconfigure(2, weight=3)  
        # changes row => smaller
        self.root.grid_rowconfigure(5, weight=1)

        # This helps ensure the main tree is larger than changes.
        self.root.grid_columnconfigure(0, weight=1)

        # Changes list
        self.xChangesFrame.grid_rowconfigure(0, weight=1)
        self.xChangesFrame.grid_columnconfigure(0, weight=1)

        # Metrics list
        self.xMetricsFrame.grid_rowconfigure(0, weight=1)
        self.xMetricsFrame.grid_columnconfigure(0, weight=1)

    ###########################################################################
    #                          Basic UI handlers
    ###########################################################################
    def setHivePath(self):
        """Set the hive path from user input."""
        sInputPath = self.xHivePathInputBox.get().strip()
        self.sHivePath = sInputPath
        messagebox.showinfo("Path Set", f"Hive path set to: {self.sHivePath}")

    def setModelPath(self):
        """Set the model path from user input."""
        sInputPath = self.xModelPathInputBox.get().strip()
        self.sModelPath = sInputPath
        messagebox.showinfo("Model Path Set", f"Model path set to: {self.sModelPath}")

    def setEntryLimit(self):
        """Set the entry limit from user input."""
        try:
            self.nEntryLimit = int(self.xEntryLimitInput.get().strip())
            messagebox.showinfo("Entry Limit Set", f"Entry limit set to: {self.nEntryLimit}")
        except ValueError:
            messagebox.showerror("Error", "Invalid entry limit. Please enter a valid number.")

    def setInterval(self):
        """Set the auto-refresh interval from user input."""
        try:
            self.nInterval = int(self.xIntervalInput.get().strip())
            messagebox.showinfo("Interval Set", f"Auto-refresh interval set to: {self.nInterval} seconds")
        except ValueError:
            messagebox.showerror("Error", "Invalid interval. Please enter a valid number.")

    def setAppendParsedCsvPath(self):
        """Set path to an existing CSV for appending parsed data."""
        sInputPath = self.xAppendParsedCsvInputBox.get().strip()
        if not sInputPath:
            self.sAppendParsedCsvPath = ''
            messagebox.showinfo("Append Parsed CSV Path", "Cleared. Will save new CSV for parsed data by default.")
            return

        if not os.path.exists(sInputPath):
            messagebox.showerror("Error", f"CSV does not exist at: {sInputPath}")
            return

        self.sAppendParsedCsvPath = sInputPath
        messagebox.showinfo("Append Parsed CSV Path Set", f"Will append parsed data to: {self.sAppendParsedCsvPath}")

    def setAppendPreprocessedCsvPath(self):
        """Set path to an existing CSV for appending preprocessed data."""
        sInputPath = self.xAppendPreprocessedCsvInputBox.get().strip()
        if not sInputPath:
            self.sAppendPreprocessedCsvPath = ''
            messagebox.showinfo("Append Preprocessed CSV Path", "Cleared. Will save new CSV for preprocessed data by default.")
            return

        if not os.path.exists(sInputPath):
            messagebox.showerror("Error", f"CSV does not exist at: {sInputPath}")
            return

        self.sAppendPreprocessedCsvPath = sInputPath
        messagebox.showinfo("Append Preprocessed CSV Path Set", f"Will append preprocessed data to: {self.sAppendPreprocessedCsvPath}")

    ###########################################################################
    #                    Helper: Append DataFrame to Existing CSV
    ###########################################################################
    def appendToExistingCsv(self, new_df: pd.DataFrame, default_csv_path: str, sAppendPath: str) -> None:
        """Attempt to append 'new_df' to 'sAppendPath' if valid & columns match; else save new."""
        if not sAppendPath:
            new_df.to_csv(default_csv_path, index=False)
            print(f"Data saved to new CSV: {default_csv_path}")
            return

        try:
            if not os.path.exists(sAppendPath):
                messagebox.showerror("Error", f"Append CSV does not exist: {sAppendPath}")
                new_df.to_csv(default_csv_path, index=False)
                print(f"Data saved to new CSV (fallback): {default_csv_path}")
                return

            existing_df = pd.read_csv(sAppendPath)
            if set(existing_df.columns) != set(new_df.columns):
                msg = (
                    "Columns in the existing CSV do not match new DataFrame columns.\n"
                    "Cannot append. Will save to a new CSV."
                )
                messagebox.showerror("Error", msg)
                new_df.to_csv(default_csv_path, index=False)
                print(f"Data saved to new CSV (columns mismatch fallback): {default_csv_path}")
                return

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(sAppendPath, index=False)
            print(f"Data appended to existing CSV: {sAppendPath}")

        except Exception as e:
            err_msg = f"Error appending to CSV ({sAppendPath}): {e}"
            print(err_msg)
            messagebox.showerror("Error", err_msg)
            new_df.to_csv(default_csv_path, index=False)
            print(f"Data saved to new CSV (exception fallback): {default_csv_path}")

    ###########################################################################
    #                       Registry Parsing & GUI Loading
    ###########################################################################
    def parseRegistry(self, sHivePath):
        """
        Parse the registry hive using regipy.
        Returns a pandas DataFrame with columns:
         ['Key', 'Depth', 'Key Size', 'Subkey Count', 'Value Count', 'Name', 'Value', 'Type'].
        """
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

            if not xData:
                print("No data parsed from registry.")
                return pd.DataFrame()

            df = pd.DataFrame(xData)
            raw_csv_path = os.path.join(os.getcwd(), "parsed_registry.csv")

            self.appendToExistingCsv(df, raw_csv_path, self.sAppendParsedCsvPath)

            return df

        except Exception as e:
            error_message = f"Failed to parse registry: {e}"
            print(error_message)
            messagebox.showerror("Error", error_message)
            return pd.DataFrame()

    def loadGUITrees(self, xData: pd.DataFrame):
        """Load parsed registry data (DataFrame) into the Treeview."""
        self.xKeyTrees.delete(*self.xKeyTrees.get_children())
        for _, row in xData.iterrows():
            self.xKeyTrees.insert(
                '',
                'end',
                values=(
                    row.get('Name', ''),
                    row.get('Value', ''),
                    row.get('Type', ''),
                    row.get('Subkey Count', ''),
                    row.get('Value Count', ''),
                    row.get('Key Size', ''),
                    row.get('Depth', '')
                )
            )

    ###########################################################################
    #                          Preprocessing
    ###########################################################################
    def categorizePath(self, path):
        """Example path categorizer."""
        if "Run" in path:
            return "Startup Path"
        elif "Services" in path:
            return "Service Path"
        elif "Internet Settings" in path:
            return "Network Path"
        else:
            return "Other Path"

    def mapType(self, value_type):
        """Group registry value types."""
        type_grouping = {
            "String":  ["REG_SZ", "REG_EXPAND_SZ", "REG_MULTI_SZ"],
            "Numeric": ["REG_DWORD", "REG_QWORD"],
            "Binary":  ["REG_BINARY"],
            "Others":  [
                "REG_NONE", "REG_LINK", "REG_RESOURCE_LIST",
                "REG_FULL_RESOURCE_DESCRIPTOR", "REG_RESOURCE_REQUIREMENTS_LIST"
            ],
        }
        for group, reg_types in type_grouping.items():
            if value_type in reg_types:
                return group
        return "Others"

    def categorizeKeyName(self, key_name):
        """Example name-based categorizer for registry values."""
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

    def preprocessValue(self, value):
        """Convert string values to length, keep numeric as is, else 0."""
        if isinstance(value, str):
            return len(value)
        elif isinstance(value, (int, float)):
            return value
        else:
            return 0

    def preprocessData(self, df: pd.DataFrame, do_random_label=True) -> pd.DataFrame:
        """
        Preprocess the parsed registry DataFrame into feature-engineered form.
        """
        if df is None or df.empty:
            print("No valid data to preprocess.")
            return pd.DataFrame()

        try:
            xDf = df.copy()
            xDf.fillna(xDf.select_dtypes(include=[np.number]).mean(), inplace=True)

            # Path Category
            xDf['Path Category'] = xDf['Key'].apply(self.categorizePath)
            path_encoded = pd.get_dummies(xDf['Path Category'], prefix='PathCategory')
            xDf = pd.concat([xDf.drop('Path Category', axis=1), path_encoded], axis=1)

            # Type Group
            xDf['Type Group'] = xDf['Type'].apply(self.mapType)
            type_group_encoded = pd.get_dummies(xDf['Type Group'], prefix='TypeGroup')
            xDf = pd.concat([xDf.drop(['Type', 'Type Group'], axis=1), type_group_encoded], axis=1)

            # Key Name Category
            xDf['Key Name Category'] = xDf['Name'].apply(self.categorizeKeyName)
            key_name_encoded = pd.get_dummies(xDf['Key Name Category'], prefix='KeyNameCategory')
            xDf = pd.concat([xDf.drop('Key Name Category', axis=1), key_name_encoded], axis=1)

            # Convert Value
            xDf['Value Processed'] = xDf['Value'].apply(self.preprocessValue)
            xDf.drop('Value', axis=1, inplace=True)

            # Scale numeric
            scaler_minmax = MinMaxScaler()
            minmax_cols = ['Depth', 'Value Count', 'Value Processed']
            xDf[minmax_cols] = scaler_minmax.fit_transform(xDf[minmax_cols])

            scaler_robust = RobustScaler()
            robust_cols = ['Key Size', 'Subkey Count']
            xDf[robust_cols] = scaler_robust.fit_transform(xDf[robust_cols])

            if do_random_label:
                xDf['Label'] = np.random.choice(['Benign', 'Malicious'], size=len(xDf))
            else:
                xDf['Label'] = 'Benign'

            xDf.drop(columns=['Key', 'Name'], inplace=True, errors='ignore')

            # Build default new CSV path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(script_dir, exist_ok=True)
            sTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sOutputCsv = os.path.join(script_dir, f"preprocessed_{sTimestamp}.csv")

            self.appendToExistingCsv(xDf, sOutputCsv, self.sAppendPreprocessedCsvPath)

            return xDf

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return pd.DataFrame()

    ###########################################################################
    #                              ML Pipeline
    ###########################################################################
    def performRFE(self, X_train, y_train, n_features=10):
        """Recursive Feature Elimination to pick top features."""
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(model, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)
            return X_train.columns[rfe.support_]
        except Exception as e:
            print(f"Error in performRFE: {e}")
            return X_train.columns

    def performGridSearch(self, X_train, y_train):
        """Hyperparameter tuning with GridSearch."""
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def performKFoldCrossValidation(self, model, X, y, n_splits=5):
        """K-Fold CV to check generalization."""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores

    def trainAndEvaluateModel(self, preprocessed_df, do_rfe=True, do_grid_search=True, do_kfold=True):
        """Train a RandomForest and evaluate using a test split, RFE, GridSearch, and K-Fold."""
        if preprocessed_df.empty or 'Label' not in preprocessed_df.columns:
            print("Cannot train. DataFrame is empty or missing 'Label'.")
            return

        X = preprocessed_df.drop(columns=['Label'])
        y = (preprocessed_df['Label'] == 'Malicious').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if do_rfe:
            selected_features = self.performRFE(X_train, y_train, n_features=10)
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

        if do_grid_search:
            model = self.performGridSearch(X_train, y_train)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Classification report as a dict so we can extract precision/recall/f1
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # We'll handle binary class = '1'
        precision_1 = report_dict['1']['precision'] if '1' in report_dict else 0.0
        recall_1    = report_dict['1']['recall']    if '1' in report_dict else 0.0
        f1_1        = report_dict['1']['f1-score']  if '1' in report_dict else 0.0

        print(f"Accuracy (threshold=0.5): {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        if do_kfold:
            scores = self.performKFoldCrossValidation(model, X, y, n_splits=5)
            print(f"Mean {5}-Fold Score: {scores.mean():.4f}")

        # AUC and threshold
        y_scores = model.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        opt_idx = np.argmin(distances)
        opt_threshold = thresholds[opt_idx]
        y_pred_opt = (y_scores >= opt_threshold).astype(int)
        opt_acc = accuracy_score(y_test, y_pred_opt)

        print(f"ROC AUC: {auc_val:.4f}")
        print(f"Optimal Threshold by Dist-to-(0,1): {opt_threshold:.4f}")
        print(f"Accuracy (threshold={opt_threshold:.4f}): {opt_acc:.4f}")
        print("Classification Report (opt threshold):\n", classification_report(y_test, y_pred_opt))

        # (1) Save the model
        joblib.dump(model, 'trained_model.joblib')
        print("Model saved: trained_model.joblib")

        # (2) Save classified CSV
        X_test_copy = X_test.copy()
        X_test_copy['Actual']    = np.where(y_test == 1, 'Malicious', 'Benign')
        X_test_copy['Pred_0.5']  = np.where(y_pred == 1, 'Malicious', 'Benign')
        X_test_copy['Pred_opt']  = np.where(y_pred_opt == 1, 'Malicious', 'Benign')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classified_csv = os.path.join(sScriptPath, f"classified_{timestamp}.csv")
        X_test_copy.to_csv(classified_csv, index=False)
        print(f"Classified results saved to {classified_csv}")

        # (3) Build metrics dict and update UI
        metrics_dict = {
            "Accuracy (0.5)": f"{acc:.4f}",
            "Precision (Class=1)": f"{precision_1:.4f}",
            "Recall (Class=1)": f"{recall_1:.4f}",
            "F1-Score (Class=1)": f"{f1_1:.4f}",
            "ROC AUC": f"{auc_val:.4f}",
            "Optimal Threshold": f"{opt_threshold:.4f}",
            "Accuracy (opt)": f"{opt_acc:.4f}",
        }
        self.updateMetricsDisplay(metrics_dict)

    ###########################################################################
    #                    Update the Metrics Treeview in the UI
    ###########################################################################
    def updateMetricsDisplay(self, metrics: dict):
        """Clear the existing metrics items, then populate the Treeview."""
        self.xMetricsList.delete(*self.xMetricsList.get_children())
        for metric_name, metric_value in metrics.items():
            self.xMetricsList.insert("", "end", values=(metric_name, metric_value))

    ###########################################################################
    #                      Refresh & Change Detection
    ###########################################################################
    def refreshMILOTIC(self):
        """Refresh data from the hive path, parse, load the GUI, check changes."""
        try:
            self.xLoadingLabel.config(text="Loading...")
            self.root.update_idletasks()

            if os.path.exists(self.sHivePath):
                sHiveType = os.path.basename(self.sHivePath).split('.')[0].lower()
                if sHiveType != self.sPreviousHiveType:
                    self.xChangesList.delete(*self.xChangesList.get_children())
                    self.xPreviousData = None
                    self.sPreviousHiveType = sHiveType

                xData = self.parseRegistry(self.sHivePath)
                self.loadGUITrees(xData)

                if self.xPreviousData is not None and not self.xPreviousData.empty:
                    self.checkChanges(self.xPreviousData, xData)

                self.xPreviousData = xData
            else:
                print(f"Hive path does not exist: {self.sHivePath}")
        finally:
            self.xLoadingLabel.config(text="")

    def toggleAutoRefreshMILOTIC(self):
        """Enable/disable auto-refresh."""
        self.bAutoRefresh = not self.bAutoRefresh
        self.xAutoRefreshButton.config(text="Disable Auto Refresh" if self.bAutoRefresh else "Enable Auto Refresh")
        if self.bAutoRefresh:
            self.autoRefreshMILOTIC()

    def autoRefreshMILOTIC(self):
        """Auto-refresh using Tkinter 'after' method."""
        if self.bAutoRefresh:
            self.refreshMILOTIC()
            self.root.after(self.nInterval * 1000, self.autoRefreshMILOTIC)

    ###########################################################################
    #                         Sorting & CSV Export
    ###########################################################################
    def sortTreeview(self, sCol, bReverse):
        """Sort items in the Treeview by column index or heading."""
        items = list(self.xKeyTrees.get_children(''))
        col_idx = list(self.xKeyTrees["columns"]).index(sCol)

        def get_val(item):
            return self.xKeyTrees.set(item, sCol)

        sorted_items = sorted(items, key=lambda i: get_val(i), reverse=bReverse)
        for idx, itm in enumerate(sorted_items):
            self.xKeyTrees.move(itm, '', idx)

        self.xKeyTrees.heading(sCol, command=lambda: self.sortTreeview(sCol, not bReverse))
        self.exportSortedCSV()

    def exportSortedCSV(self):
        """Export the data currently in the Treeview to a CSV."""
        columns = self.xKeyTrees["columns"]
        data = []
        for item in self.xKeyTrees.get_children():
            row_vals = self.xKeyTrees.item(item, 'values')
            data.append(row_vals)

        df = pd.DataFrame(data, columns=columns)
        df.dropna(axis=1, how='all', inplace=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(sScriptPath, f"snapshot_sorted_{timestamp}.csv")
        df.to_csv(output_csv, index=False)
        messagebox.showinfo("Export Complete", f"Sorted data exported to: {output_csv}")

    ###########################################################################
    #                     Change Detection & Export
    ###########################################################################
    def compareRegistrySnapshots(self, old_df: pd.DataFrame, new_df: pd.DataFrame):
        """Compare two snapshots of registry data for additions or removals."""
        if old_df.empty or new_df.empty:
            return []

        old_set = {f"{r.Key}|{r.Name}|{r.Value}" for _, r in old_df.iterrows()}
        new_set = {f"{r.Key}|{r.Name}|{r.Value}" for _, r in new_df.iterrows()}

        added = new_set - old_set
        removed = old_set - new_set

        xChanges = []
        for a in added:
            xChanges.append({"Action": "Added", "Description": a})
        for r in removed:
            xChanges.append({"Action": "Removed", "Description": r})
        return xChanges

    def checkChanges(self, old_df, new_df):
        """Populates the changes list if there's any difference between old/new snapshots."""
        xChanges = self.compareRegistrySnapshots(old_df, new_df)
        self.xChangesList.delete(*self.xChangesList.get_children())
        for chg in xChanges:
            self.xChangesList.insert('', 'end', values=(chg['Action'], chg['Description']))

        if xChanges:
            df_changes = pd.DataFrame(xChanges)
            sTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sOutputCsv = os.path.join(sScriptPath, f"changes_{sTimestamp}.csv")
            df_changes.to_csv(sOutputCsv, index=False)
            print(f"Changes exported to: {sOutputCsv}")

    ###########################################################################
    #                     High-Level ML Execution
    ###########################################################################
    def executeMLProcess(self):
        """Run entire pipeline: parse => preprocess => train/eval model."""
        try:
            # Check hive path
            self.sHivePath = self.xHivePathInputBox.get().strip()
            if not self.sHivePath:
                raise ValueError("Hive path is empty.")

            print("Parsing registry data...")
            df = self.parseRegistry(self.sHivePath)
            if df.empty:
                raise ValueError("Parsed data is empty.")

            print("Preprocessing data...")
            preprocessed_df = self.preprocessData(df, do_random_label=True)
            if preprocessed_df.empty:
                raise ValueError("Preprocessed data is empty.")

            print("Training and evaluating model...")
            self.trainAndEvaluateModel(preprocessed_df)

            messagebox.showinfo("ML Process Complete", "The ML process has successfully finished!")

        except Exception as e:
            error_message = f"Error in executeMLProcess: {e}"
            print(error_message)
            messagebox.showerror("Error", error_message)


###############################################################################
#                               MAIN ENTRY POINT
###############################################################################
def main():
    root = tk.Tk()
    app = MILOTIC(root)
    root.mainloop()

if __name__ == "__main__":
    main()
