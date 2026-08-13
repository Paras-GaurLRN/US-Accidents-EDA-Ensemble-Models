from sklearn.base import (BaseEstimator, TransformerMixin, clone)
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import ArbitraryOutlierCapper
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class AnomalyCleaner(BaseEstimator, TransformerMixin):
    '''
    DOCSTRING

    Pretty simple transformer, drops all data that is not within the range.
    We may choose to cap data instead by setting 'drop' to False.

    'missing_values' is used only if we are capping. Ignores by default, the user may change to Raise.

    Use np.inf or -np.inf to indicate that no upper or lower bound should be applied.

    NOTE: Missing Values are untouched if we drop the Anomalous rows

    PARAMETERS:
        > value_limits = A dictionary of form {column_name:(min,max),}
        > drop = A boolean specifying if we drop Anomalous rows or cap the data instead
        > copy = Whether to return a copy of the dataframe or perform changes in-place
        > missing_values = Used if we cap values, 'ignore' by default. May be changed to 'raise'
    ''' 

    
    DEFAULT_LIMITS = {
        "Temperature(F)": (-60, 130),
        "Humidity(%)": (1, 100),
        "Pressure(in)": (20, 32.5),
        "Visibility(mi)": (0, 30),
        "Wind_Speed(mph)": (0, 150),
    }
    
    @staticmethod
    def _validate(value_limits):
        for col, (low,high) in value_limits.items():
            if low > high:
                raise ValueError(f"Lower Bound > Upper Bound for Key - {col}")
    
    def __init__(self,*,
                 value_limits = None,
                 drop = True,
                 copy = False,
                 missing_values='ignore'):
               
        self.value_limits = value_limits
        self.drop = drop
        self.copy = copy
        self.missing_values = missing_values

    def fit(self, X, y=None):
        self._limits = (
            self.DEFAULT_LIMITS 
            if self.value_limits is None
            else self.value_limits
        )
        
        missing = set(self._limits) - set(X.columns)
        
        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")

        type(self)._validate(self._limits)
        return self

    def transform(self, X):
        check_is_fitted(self,"_limits")

        missing = set(self._limits) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")
        
        if self.copy: X = X.copy()

        invalid = np.zeros(len(X), dtype=bool)

        if self.drop:
            for column, (low, high) in self._limits.items():
    
                invalid |= (
                    X[column].notna()
                    &
                    ~X[column].between(low, high)
                )
    
            return X.loc[~invalid].reset_index(drop=True)
        
    
        capper = ArbitraryOutlierCapper(
            max_capping_dict = {col:high for col, (_,high) in self._limits.items()},
            min_capping_dict = {col:low for col, (low,_) in self._limits.items()},
            missing_values = self.missing_values
        )

        return capper.fit_transform(X)

class DateTimeFeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    DOCSTRING

    This Transformer is to convert DateTimeFeatures into a more useful form.
    New columns added are

    PARAMETERS:
        > keep_end_features = Whether to keep accident End related features
        > filter_invalid_dates = Whether to filter invalid dates or now
        > max_resolution_days = Maximum days allowed for accident resolution, any value more than that will be marked invalid
        > copy = Whether to return a copy of the dataframe or perform changes in-place
    '''
    
    DEFAULT_RENAME_MAP = {
            "Start_Time_day_of_year": "Accident Day",
            "Start_Time_hour": "Accident Timing",
            "Start_Time_year": "Accident Year",
            "Start_Time_weekend": "Weekend",
            "Start_Time_leap_year": "Accident Leap Year",
            "End_Time_day_of_year": "End Day",
            "End_Time_hour": "Resolution Time",
            "End_Time_year": "End Year",
            "End_Time_weekend": "End Weekend",
            "End_Time_leap_year": "End Leap Year",
        }

    def _validate_dates(self, X):

        valid = pd.Series(True, index=X.index)

        valid &= X["Year Difference"].between(0, 1)

        crossed_year = X["Year Difference"] == 1

        valid &= (
            ~crossed_year
            |
            (X["End Day"] < X["Accident Day"])
        )

        valid &= X["Days To Resolve"] >= 0

        valid &= (
            X["Days To Resolve"]
            <= self.max_resolution_days
        )

        return valid
    
    def __init__(self,*,
                keep_end_features = False,
                filter_invalid_dates = True,
                max_resolution_days = 1,
                copy = False):
        if max_resolution_days < 0: raise ValueError("max_resolution_days must be non-negative.")
        
        self.keep_end_features = keep_end_features
        self.filter_invalid_dates = filter_invalid_dates
        self.max_resolution_days = max_resolution_days
        self.copy = copy
        

    def fit(self, X, y=None):
        missing = set(["Start_Time","End_Time"]) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")

        if self.copy: X = X.copy()
        
        X["Start_Time"] = pd.to_datetime(X["Start_Time"], format='mixed')
        X["End_Time"] = pd.to_datetime(X["End_Time"], format='mixed')
        
        self._datetime_features = DatetimeFeatures(
            variables=["Start_Time", "End_Time"],
            features_to_extract=[
                "day_of_year",
                "hour",
                "year",
                "weekend",
                "leap_year",
            ],
            drop_original=False,
        )
        
        self._datetime_features.fit(X)

        return self

    def transform(self, X):
        check_is_fitted(self,'_datetime_features')

        missing = set(["Start_Time","End_Time"]) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")
        
        if self.copy: X = X.copy()
            
        X["Start_Time"] = pd.to_datetime(X["Start_Time"], format='mixed')
        X["End_Time"] = pd.to_datetime(X["End_Time"], format='mixed')
        
        X = self._datetime_features.transform(X)

        X.rename(columns=self.DEFAULT_RENAME_MAP, inplace=True)
        
        X["Days To Resolve"] = (
            X["End_Time"] - X["Start_Time"]
        ).dt.total_seconds() / 86400

        X["Year Difference"] = (
            X["End Year"] - X["Accident Year"]
        )

        if self.filter_invalid_dates: 
            valid = self._validate_dates(X)
            X = X.loc[valid].copy()
            
            X.drop(
                columns=[
                    "Days To Resolve",
                    "Year Difference",
                ],
                inplace=True,
                errors="ignore",
            )
            
        X.drop(
            columns=[
                "Accident Leap Year",
                "End Leap Year",
            ],
            inplace=True,
            errors="ignore",
        )

        if not self.keep_end_features:

            X.drop(
                columns=[
                    "End Day",
                    "Resolution Time",
                    "End Weekend",
                    "End Year",
                ],
                inplace=True,
                errors="ignore",
            )

        X.drop(
            columns=[
                "Start_Time",
                "End_Time",
            ],
            inplace=True,
            errors="ignore",
        )

        return X

class Illuminator(BaseEstimator,TransformerMixin):
    '''
    DOCSTRING

    This Transformer adds the Illumination column and optionally drops the other columns.

    NOTE: It is recommended to drop the other 4 columns if Illumination is created,
          This is recommended as Illumination summarizes the four twilight indicators into a single ordered categorical feature,
          It reduces four correlated features into a single ordered categorical feature.

    PARAMETERS:
        > keep_others = If the 4 processed columns are to be kept or dropped
        > mark_invalid = If to set values to 'Invalid' if order is invalid, else marked 'Missing'
        > copy = Whether to return a copy of the dataframe or perform changes in-place
    '''
    
    DEFAULT_ILLUMINATION_ORDER = {
        0 : 'Night',
        1 : 'Astronomical Twilight',
        2 : 'Nautical Twilight',
        3 : 'Civil Twilight',
        4 : 'Day',
        'Miscellaneous' : ('Missing','Invalid')
    }

    DEFAULT_LIGHT_COLUMNS = ["Sunrise_Sunset","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight",]
    
    def __init__(self,*,
                 keep_others = False,
                 mark_invalid = True,
                 copy = False):
        
        self.keep_others = keep_others
        self.mark_invalid = mark_invalid
        self.copy = copy

    def fit(self, X, y=None):
        missing = set(self.DEFAULT_LIGHT_COLUMNS) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")

        return self

    def transform(self, X):
        missing = set(self.DEFAULT_LIGHT_COLUMNS) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")
        
        if self.copy: X = X.copy()

        X['Illumination'] = np.select(
            [
                X[self.DEFAULT_LIGHT_COLUMNS].isna().all(axis=1),
                
                (
                    (X['Sunrise_Sunset'] == 'Day') &
                    (X['Civil_Twilight'] == 'Day') &
                    (X['Nautical_Twilight'] == 'Day') &
                    (X['Astronomical_Twilight'] == 'Day')
                ),
        
                (
                    (X['Sunrise_Sunset'] == 'Night') &
                    (X['Civil_Twilight'] == 'Day') &
                    (X['Nautical_Twilight'] == 'Day') &
                    (X['Astronomical_Twilight'] == 'Day')
                ),
        
                (
                    (X['Sunrise_Sunset'] == 'Night') &
                    (X['Civil_Twilight'] == 'Night') &
                    (X['Nautical_Twilight'] == 'Day') &
                    (X['Astronomical_Twilight'] == 'Day')
                ),
        
                (
                    (X['Sunrise_Sunset'] == 'Night') &
                    (X['Civil_Twilight'] == 'Night') &
                    (X['Nautical_Twilight'] == 'Night') &
                    (X['Astronomical_Twilight'] == 'Day')
                ),
        
                (
                    (X['Sunrise_Sunset'] == 'Night') &
                    (X['Civil_Twilight'] == 'Night') &
                    (X['Nautical_Twilight'] == 'Night') &
                    (X['Astronomical_Twilight'] == 'Night')
                )
            ],
            [
                'Missing',
                'Day',
                'Civil Twilight',
                'Nautical Twilight',
                'Astronomical Twilight',
                'Night'
            ],
        
            default=('Invalid' if self.mark_invalid else 'Missing')
        )

        if not self.keep_others:
            X.drop(
                columns=self.DEFAULT_LIGHT_COLUMNS,
                inplace=True,
                errors="ignore",
            )

        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    DOCSTRING

    This Transformer Basically drops the features not needed, and has a default list as many features were not required repreatedly.

    PARAMETERS:
        > columns = List of columns to drop, not validated for convenience purposes
        > add_columns = Whether to add the columns to default list or use the passed list as the total list
        > copy =  Whether to return a copy of the dataframe or perform changes in-place
    """
    
    DEFAULT_COLUMNS = [
        "Start_Lat",
        "Start_Lng",
        "End_Lat",
        "End_Lng",
        "Distance(mi)",
        "Street",
        "City",
        "Country",
        "Timezone",
        "Description",
        "Zipcode",
        "Airport_Code",
        "Weather_Timestamp",
        "Wind_Chill(F)",
        "Precipitation(in)",
    ]
    
    def __init__(self,*,
                 columns = None,
                 add_columns = False,
                 copy = False):
        
        if (columns is None) and (add_columns):
            raise ValueError("Specify Columns if Columns are to be added")

        if columns is not None:
            columns = list(columns)
        
        self.copy = copy
        self.add_columns = add_columns
        self.columns = (self.DEFAULT_COLUMNS if (columns is None) else columns) if (not self.add_columns) else list(set(self.DEFAULT_COLUMNS + list(columns)))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.copy: X = X.copy()

        return X.drop(
            columns=self.columns,
            errors = "ignore",
        )

def make_preprocessing_pipe(*,
                            # WeatherAnomalyCleaner Parameters
                            value_limits = None,
                            drop_anomalous_weather_data = True,
                            weather_missing_values = "ignore",
                            # DateTimeFeatureEngineer Parameters
                            datetime_keep_end_features = False,
                            filter_invalid_dates = True,
                            max_resolution_days = 1,
                            # Illuminator Parameters
                            illuminator_keep_others = False,
                            illuminator_mark_invalid = True,
                            # ColumnDropper Parameters
                            columns_to_drop = None,
                            append_columns_to_drop = False,
                            # Pipeline Parameters
                            memory = None,
                            verbose = False,
                            copy = False):
    
    return Pipeline([
        ('weather_anomaly_cleaner',AnomalyCleaner(value_limits=value_limits,drop=drop_anomalous_weather_data,missing_values=weather_missing_values,copy=copy)),
        ('datetime_feature_engineer',DateTimeFeatureEngineer(keep_end_features=datetime_keep_end_features,filter_invalid_dates=filter_invalid_dates,max_resolution_days=max_resolution_days,copy=copy)),
        ('illuminator',Illuminator(keep_others=illuminator_keep_others,mark_invalid=illuminator_mark_invalid,copy=copy)),
        ('column_dropper',ColumnDropper(columns=columns_to_drop,add_columns=append_columns_to_drop,copy=copy))
    ],
    memory = memory,
    verbose = verbose)