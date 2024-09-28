import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from .config import *

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def _get_period_day(self, date: str) -> str:
        """
        Helper function to categorize time of day.

        Args:
            date (str): Input datetime.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime(MORNING_MIN, '%H:%M').time()
        morning_max = datetime.strptime(MORNING_MAX, '%H:%M').time()
        afternoon_min = datetime.strptime(AFTERNOON_MIN, '%H:%M').time()
        afternoon_max = datetime.strptime(AFTERNOON_MAX, '%H:%M').time()
        evening_min = datetime.strptime(EVENING_MAX, '%H:%M').time()
        evening_max = datetime.strptime(EVENING_MIN, '%H:%M').time()
        night_min = datetime.strptime(NIGHT_MIN, '%H:%M').time()
        night_max = datetime.strptime(NIGHT_MAX, '%H:%M').time()

        if date_time >= morning_min and date_time <= morning_max:
            return PeriodDay.MORNING
        elif date_time >= afternoon_min and date_time <= afternoon_max:
            return PeriodDay.AFTERNOON
        elif (date_time >= evening_min and date_time <= evening_max) or (date_time >= night_min and date_time <= night_max):
            return PeriodDay.NIGHT

    def _is_high_season(self, fecha: str) -> int:
        """
        Helper function to identify high season periods.
        
        Args:
            fecha (str): Input datetime.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime(HIGH_SEASON_RANGE1_MIN, '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime(HIGH_SEASON_RANGE1_MAX, '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime(HIGH_SEASON_RANGE2_MIN, '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime(HIGH_SEASON_RANGE2_MAX, '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime(HIGH_SEASON_RANGE3_MIN, '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime(HIGH_SEASON_RANGE3_MAX, '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime(HIGH_SEASON_RANGE4_MIN, '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime(HIGH_SEASON_RANGE4_MAX, '%d-%b').replace(year=fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calculate the minute difference between two date columns.

        Args:
            row (pd.Series): Row of the dataset.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff
    
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        
        
        # Generate 'delay' column based on a threshold of 15 minutes
        
        # Feature encoding
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)[TOP_10_FEATURES]
        
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data['delay'] = np.where(data['min_diff'] > DELAY_THRESHOLD_IN_MINUTES, 1, 0)
            return features, data[[target_column]]
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Balance Class Weights
        target = target.to_numpy()
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        
        # Train Logistic Regression with balanced class weights
        self._model = LogisticRegression(class_weight={1: n_y0 / len(target), 0: n_y1 / len(target)})
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return [1 if y_pred > THRESHOLD_PREDICT else 0 for y_pred in self._model.predict(features)]

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Evaluate the trained model."""
        y_preds = self._model.predict(features)
        print(confusion_matrix(target, y_preds))
        print(classification_report(target, y_preds))