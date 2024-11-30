import pandas as pd 
import re

from sklearn.preprocessing import OneHotEncoder, StandardScaler

class NesuDataPreparation: 
    
    target_var = 'days_to_next'
    
    def prepare_data_for_training(self, dataset): 
        
        # 1. Clean 
        df = self.__clean_data(dataset)
        
        # 2. Training examples creation: Interpolation
        df = self.__interpolate_days(df)
        
        # 3. Feature Engineering
        df = self.__engineer_features(df)
        
        # 4. Filter outliers
        df = self.__filter_rows(df)
        
        # 5. Select Features
        df = self.__feature_selection(df)
        
        # 6. Encode & Scale
        encoding_result = self.__one_hot_encode(df)
        
        df = encoding_result['dataset']
        
        scaling_result = self.__scale(df)
        
        df = scaling_result['dataset']
        
        return {
            'dataset': df, 
            'trained_encoders': encoding_result['trained_encoders'], 
            'trained_scalers': scaling_result['trained_scalers']
        }
        
    
    def __clean_data(self, df):
        """Cleans the data as a first step of the preparation process"""

        # Remove poor quality data
        to_remove = ['normal', 'smøger', 'matas', 'chews', 'dettori', 'mobilepay', 'mobile', 'bager', 'supermarco', 'friends', 'nespresso', 'vogue', 'lindt', 'tina', 'coffee', '7', 'eleven', 'mor', 'frederik', 'zoo', 'zoobidoo', 'hundemad', 'vegas', 'shampoo', 'esselunga']
        
        pattern = '|'.join([re.escape(word) for word in to_remove])  # Escapes special characters
        
        df = df[~df['description'].str.lower().str.contains(pattern, na=False)]

        # Remove low spend rows
        df = df[df['amountInEuro'] > 20]

        return df

    def __build_monthly_spend(self, df):
        """Builds a monthly spend"""

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        df = df.groupby(by=['year', 'month'])['amountInEuro'].agg(
            total_spend='sum',
            average_spend='mean'
        ).reset_index()

        return df
        
    def __interpolate_days(self, dataset: pd.DataFrame): 
        """Interpolates the days, creating intermediary records for training."""

        # Create averages
        spend_df = self.__build_monthly_spend(dataset)

        # Create the full range of dates
        date_range = pd.date_range(start=dataset['date'].min(), end=dataset['date'].max())

        # Initialize the new DataFrame
        df = pd.DataFrame({'date': date_range})

        # Compute days_since_last
        df['days_since_last'] = df['date'].apply(
            lambda d: (d - dataset[dataset['date'] < d]['date'].max()).days if not dataset[dataset['date'] < d].empty 
            else 0
        )

        # Compute last_day_of_week
        df['last_dow'] = df['date'].apply(
            lambda d: (dataset[dataset['date'] < d]['date'].max()).day_name() if not dataset[dataset['date'] < d].empty 
            else None
        )

        # Compute days_to_next
        df[self.target_var] = df['date'].apply(
            lambda d: 0 if d in dataset['date'].values
            else (dataset[dataset['date'] > d]['date'].min() - d).days if not dataset[dataset['date'] > d].empty 
            else 0
        )
        
        # Compute last_amount_spent
        df['last_amount_spent'] = df['date'].apply(
            lambda d: dataset[dataset['date'] < d].iloc[-1]['amountInEuro'] if not dataset[dataset['date'] < d].empty 
            else None
        )
        
        # Who went last to the supermarket
        df['last_shopper'] = df['date'].apply(
            lambda d: dataset[dataset['date'] < d].iloc[-1]['user'] if not dataset[dataset['date'] < d].empty 
            else None
        )
        
        # Find the last average and total spend
        df['last_month_total_spend'] = df['date'].apply(
            lambda d: spend_df[(spend_df['year'] == d.year) & (spend_df['month'] == d.month-1)].iloc[0]['total_spend'] if d.month > 1 and not spend_df[(spend_df['year'] == d.year) & (spend_df['month'] == d.month-1)].empty
            else spend_df[(spend_df['year'] == d.year-1) & (spend_df['month'] == 12)].iloc[0]['total_spend'] if d.month == 1 and not spend_df[(spend_df['year'] == d.year-1) & (spend_df['month'] == 12)].empty
            else None
        )
        df['last_month_avg_spend'] = df['date'].apply(
            lambda d: spend_df[(spend_df['year'] == d.year) & (spend_df['month'] == d.month-1)].iloc[0]['average_spend'] if d.month > 1 and not spend_df[(spend_df['year'] == d.year) & (spend_df['month'] == d.month-1)].empty
            else spend_df[(spend_df['year'] == d.year-1) & (spend_df['month'] == 12)].iloc[0]['average_spend'] if d.month == 1 and not spend_df[(spend_df['year'] == d.year-1) & (spend_df['month'] == 12)].empty
            else None
        )

        # Remove NaN
        df.dropna(inplace=True)

        return df
    
    def __engineer_features(self, df):

        # Month of the year
        df['month'] = df['date'].dt.month

        # Day of the week
        df['dow'] = df['date'].dt.day_name()

        # Location: Randersgade, Havneholvem, Solrod
        df['location'] = df['date'].apply(
            lambda x: 'Solrod' if x.year >= 2021 or (x.year == 2020 and x.month >= 8)
            else 'Randersgade' if x.year == 2018 and x.month < 4
            else 'Haveholmen'
        )

        # Child
        df['child'] = df['date'].apply(lambda d: 1 if d.year > 2022 or (d.year == 2022 and d.month > 8) else 0)

        # Ratio of last spend to last_month spend
        df['spend_ratio'] = df['last_amount_spent'] / df['last_month_total_spend']
        df['avg_spend_ratio'] = df['last_amount_spent'] / df['last_month_avg_spend']

        return df
    
    def __filter_rows(self, df): 

        df = df[df[self.target_var] < 10]
        df = df[df['days_since_last'] < 10]

        return df 

    def __get_selected_features(self): 
        return {
            'categorical': ['month', 'last_dow', 'dow', 'location', 'child', 'last_shopper'],
            'numerical': ['days_since_last', 'last_amount_spent', 'avg_spend_ratio']
        }

    def __get_categorical_feature_names(self):
        return self.__get_selected_features()['categorical']

    def __get_numerical_feature_names(self): 
        return self.__get_selected_features()['numerical']
    
    def __feature_selection(self, dataset): 
        features = self.__get_selected_features()
        return dataset[features['categorical'] + features['numerical'] + [self.target_var]]

    def __one_hot_encode(self, df, trained_encoders=None): 

        # 1. Get the categorical features
        cat_features = self.__get_categorical_feature_names()

        # Maintain a dict of encoders
        encoders = {}

        # 2. Encode each feature
        for feature_name in cat_features: 

            if trained_encoders is None: 
                # New Encoder
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                # Fit Encoder
                encoded_data = encoder.fit_transform(df[[feature_name]])
            else:
                encoder = trained_encoders[feature_name]
                # Transform
                encoded_data = encoder.transform(df[[feature_name]])

            encoded_feature = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

            # Add the new features
            df = pd.concat([df.reset_index(drop=True), encoded_feature.reset_index(drop=True)], axis=1)

            # Save the encoder
            encoders[feature_name] = encoder

        df = df.drop(columns=cat_features)

        return {
            'dataset': df, 
            'trained_encoders': encoders
        }

    def __scale(self, df, trained_scalers=None): 

        # 1. Get the numerical features
        numerical_features = self.__get_numerical_feature_names()

        # Save the scalers
        scalers = {}

        # 2. Scale each feature 
        for feature_name in numerical_features: 

            # Create the scaler
            if trained_scalers is None:
                scaler = StandardScaler()
                #Fit 
                df[feature_name] = scaler.fit_transform(df[[feature_name]])
            else:
                scaler = trained_scalers[feature_name]
                # Transform 
                df[feature_name] = scaler.transform(df[[feature_name]])

            # Save scaler 
            scalers[feature_name] = scaler

        return {
            'dataset': df,
            'trained_scalers': scalers
        }