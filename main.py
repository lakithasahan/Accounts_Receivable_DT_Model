import pickle

import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

pd.set_option('display.max_columns', 500)
labelencoder = LabelEncoder()


class AR_class():

    def __init__(self, Settled_Date_Column_Name, Invoice_Date_Column_Name, Invoice_Settled_Flag_Column_Name,
                 Customer_Names_Column_Name, Invoice_Amount_Column_Name):
        df = dd.read_csv('data/haleys-ac-*.csv', dtype={'Invoice Reference': 'object'})
        df = df.compute()
        print(df)
        self.input_data, self.target_data = self.preprocessed_df = self.prepossessing_setup(df,
                                                                                            Settled_Date_Column_Name,
                                                                                            Invoice_Date_Column_Name,
                                                                                            Invoice_Settled_Flag_Column_Name,
                                                                                            Customer_Names_Column_Name,
                                                                                            Invoice_Amount_Column_Name)
        self.input_data = self.scaling(self.input_data)

    def accuracy_check(self, y_pred, y_test):
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("F1 Accuracy:", f1_score(y_test, y_pred, average='weighted'))
        print(y_pred)
        print(y_test)

    def splitting_days_late_to_stages(self, days_late_list):
        stage_list = []
        for i in range(len(days_late_list)):
            days_late = days_late_list[i]

            if days_late < 7:
                stage_list.append(0)
            elif days_late > 7 and days_late < 30:
                stage_list.append(1)
            elif days_late > 30 and days_late < 60:
                stage_list.append(2)
            elif days_late > 60 and days_late < 90:
                stage_list.append(3)
            else:
                stage_list.append(4)
        return stage_list

    def data_arrange_for_model(self, df):
        a = df[['Month']].values.tolist()
        b = df[['Day']].values.tolist()
        c = df[['Customer_Name_Encoded']].values.tolist()
        d = df[['Company']].values.tolist()
        f = pd.Series(df.iloc[:, len(df.columns) - 1].values)
        f = list(f.dt.dayofweek.values)

        input_data = []
        for j in range(len(df)):
            input_data.append([a[j][0], b[j][0], f[j], c[j][0], d[j][0]])

        print(df)
        target_data = np.ravel(df['Payment_Stage'].values.tolist())
        return input_data, target_data

    def scaling(self, input_data):

        pca = PCA(n_components=4)
        input_data = pca.fit_transform(input_data)
        filename = 'models/PCA_obj.sav'
        pickle.dump(pca, open(filename, 'wb'))

        scaler = MinMaxScaler()
        scaler.fit(input_data)
        input_data = scaler.transform(input_data)
        filename = 'models/MinMaxscale_obj.sav'
        pickle.dump(scaler, open(filename, 'wb'))

        return input_data

    def prepossessing_setup(self, df, Settled_Date_Column_Name, Invoice_Date_Column_Name,
                            Invoice_Settled_Flag_Column_Name, Customer_Names_Column_Name, Invoice_Amount_Column_Name):
        print('Prepossessing setup')
        df_ = df.copy()

        # Initial Step for haley data model
        for x in range(len(df_)):
            if df_[Settled_Date_Column_Name].iloc[x] == '00.00.0000':
                df_[Settled_Date_Column_Name].iloc[x] = df_[Invoice_Date_Column_Name].iloc[x]

        df_[Invoice_Date_Column_Name] = pd.to_datetime(df_[Invoice_Date_Column_Name], dayfirst=True)
        df_[Settled_Date_Column_Name] = pd.to_datetime(df_[Settled_Date_Column_Name], dayfirst=True)

        # Calculating Late Days
        df_['Days_Late'] = df_[Settled_Date_Column_Name] - df_[Invoice_Date_Column_Name]
        df_['Days_Late'] = df_['Days_Late'] / np.timedelta64(1, 'D')

        # thresholding
        df_ = df_[df_['Days_Late'] >= 0]
        df_ = df_[df_[Invoice_Amount_Column_Name] > 0]
        df_ = df_[df_[Invoice_Settled_Flag_Column_Name] == 'X']
        print(df_)

        # Dtype Change
        df_[Customer_Names_Column_Name] = df_[Customer_Names_Column_Name].astype(str)

        # Encoding
        label_encoder = LabelEncoder()
        Customer_Names_Encoded = label_encoder.fit_transform(df_[Customer_Names_Column_Name])
        df_['Customer_Name_Encoded'] = Customer_Names_Encoded.tolist()
        filename = 'models/Label_encoder_obj.sav'
        pickle.dump(label_encoder, open(filename, 'wb'))



        df_['Payment_Stage'] = self.splitting_days_late_to_stages(df_['Days_Late'].tolist())
        df_['Month'] = df_[Invoice_Date_Column_Name].dt.strftime("%m").astype(int)
        df_['Day'] = df_[Invoice_Date_Column_Name].dt.strftime("%d").astype(int)

        # Null Value Drop
        df_ = df_[['Company', 'Customer_Name_Encoded', 'Month', 'Day', 'Payment_Stage', Invoice_Amount_Column_Name,Invoice_Date_Column_Name]]
        df_ = df_.dropna()
        df_ = df_.sort_values(by=Invoice_Date_Column_Name)

        df_ = df_.reset_index()


        input_data, target_data = self.data_arrange_for_model(df_)

        print(df_)

        return input_data, target_data


    def model_setup(self):
        X_train, X_test, y_train, y_test = train_test_split(self.input_data,self.target_data, test_size=0.1,
                                                            random_state=42,
                                                            shuffle=True)
        clf = RandomForestClassifier()
        parameters = {'criterion': ('gini', 'entropy'), 'n_estimators': [100, 150], 'max_depth': [1, 1000],
                      'max_features': ('auto', 'sqrt', 'log2')}
        grid_clf_acc = GridSearchCV(clf, param_grid=parameters, scoring='f1_weighted', n_jobs=-1, cv=10)
        grid_clf_acc.fit(X_train, y_train)
        print(grid_clf_acc.best_params_)
        # Predict values based on new parameters
        y_pred = grid_clf_acc.predict(X_test)

        y_pred_train = grid_clf_acc.predict(X_train)
        # tree.plot_tree(clf)
        print('train accuracy')
        obj.accuracy_check(y_pred_train, y_train)
        print('test_accuracy')
        obj.accuracy_check(y_pred, y_test)

        return grid_clf_acc,X_test, y_test


obj = AR_class('Rec Date', 'Due Date', 'Is Invoice Settled', 'Name', 'Inv Amount(LC)')
grid_clf_acc,X_test, y_test=obj.model_setup()



filename = 'models/AR_model.sav'
pickle.dump(grid_clf_acc, open(filename, 'wb'))

filename = 'models/AR_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
