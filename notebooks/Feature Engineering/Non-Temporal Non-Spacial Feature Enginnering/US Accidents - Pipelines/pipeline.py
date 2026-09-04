from warnings import warn
from sklearn.base import (BaseEstimator, TransformerMixin, clone)
from sklearn.utils._param_validation import StrOptions
from imblearn.base import BaseSampler
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import ArbitraryOutlierCapper
from imblearn.pipeline import Pipeline as IMBPipe
import pandas as pd
import numpy as np

class AnomalyCleaner(BaseSampler, TransformerMixin):
    '''
    DOCSTRING

    A 2 in 1 Sampler x Transformer,
    During training:
        drop_when_training=True  -> remove anomalous observations
        drop_when_training=False -> cap anomalous observations

    During inference:
        transform() always caps anomalous observations.
        'missing_values' is used only if we are capping. Ignores by default, the user may change to Raise.

    Use np.inf or -np.inf to indicate that no upper or lower bound should be applied.

    NOTE: Missing Values are untouched if we drop the Anomalous rows
    NOTE: This Sampler x Transformer uses a strict 'fit(X) -> fit_resample(X,y) -> transform(X)' non-standard API to ensure consistency.
          fit() initializes the anomaly limits and internal capper but does not mark the cleaner as trained;
          fit_resample() must be called before transform() is permitted.

    * When used inside an imblearn Pipeline, the estimator must be explicitly
    fitted before being supplied to the pipeline, because fit_resample()
    intentionally does not call fit().
    
    * Please read the error messages and warnings for detailed implementation overview.

    PARAMETERS:
        > value_limits = A dictionary of form {column_name:(min,max),}
        > drop_when_training = A boolean specifying if we drop Anomalous rows or cap the data instead
        > copy = Whether to return a copy of the dataframe or perform changes in-place
        > missing_values = Used if we cap values, 'ignore' by default. May be changed to 'raise'
        > grid_mode = To Suppress warning when used in a gridsearch
    ''' 

    # Design rationale: The non-standard lifecycle is intentional. It separates capper initialization from training-time resampling, 
    # and ensures that the same cleaner can operate as a sampler during training and as a capper during inference.

    _parameter_constraints = {
        "value_limits": [dict, None],
        "drop_when_training": ["boolean"],
        "copy": ["boolean"],
        "missing_values": [StrOptions({"ignore", "raise"})],
        "grid_mode" : ["boolean"]
    }
    
    DEFAULT_LIMITS = {
        "Temperature(F)": (-60, 130),
        "Humidity(%)": (1, 100),
        "Pressure(in)": (20, 32.5),
        "Visibility(mi)": (0, 30),
        "Wind_Speed(mph)": (0, 150),
        "Precipitation(in)" : (0, np.inf)
    }

    def __getattribute__(self, name):
        if name in ("transform", "fit_transform"):
            import inspect
    
            for frame in inspect.stack():
                module = frame.frame.f_globals.get("__name__", "")
    
                if module == "imblearn.pipeline":
                    raise AttributeError(
                        "AnomalyCleaner.transform is hidden from "
                        "imblearn.Pipeline so that AnomalyCleaner is "
                        "treated as a sampler."
                    )
    
        return super().__getattribute__(name)
    
    @staticmethod
    def _validate(value_limits):
        for col, (low,high) in value_limits.items():
            if low > high:
                raise ValueError(f"Lower Bound > Upper Bound for Key - {col}")
    
    def __init__(self,*,
                 value_limits = None,
                 drop_when_training = False,
                 copy = False,
                 missing_values='ignore',
                 grid_mode = False):
        self.grid_mode = grid_mode
        if not self.grid_mode:
            warn(
                "AnomalyCleaner has a dual sampler/transformer interface and is "
                "customized for compatibility with imbalanced-learn's Pipeline API. "
                "When used inside an imblearn Pipeline, AnomalyCleaner is treated as "
                "a sampler: it participates during fitting via fit_resample(), but "
                "samplers are skipped during prediction/inference. Consequently, "
                "transform()-based capping of anomalous observations will not be "
                "applied automatically by pipeline.predict(). To apply inference-time "
                "capping, either use AnomalyCleaner.transform() independently before "
                "passing the data to the remaining pipeline steps, or treat "
                "validation and capping of prediction data as the responsibility "
                "of the calling API.",
                UserWarning
            )
               
        self.value_limits = value_limits
        self.drop_when_training = drop_when_training
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

        self._capper = ArbitraryOutlierCapper(
            max_capping_dict = {col:high for col, (_,high) in self._limits.items()},
            min_capping_dict = {col:low for col, (low,_) in self._limits.items()},
            missing_values = self.missing_values
        )
        self._capper.fit(X, y)
        
        return self

    def fit_resample(self, X, y=None):
        return self._fit_resample(X, y)
    
    def _fit_resample(self, X, y=None):
        check_is_fitted(self,["_limits","_capper"],
                        msg="Sampler x Transformer, %(name)s must be fitted before resampling. Please run AnomalyCleaner.fit() "
                        "If AnomalyCleaner raised this issue when in an imbalanced-learn Pipeline, please run fit() once before using "
                        "it in the Pipeline. This API explictly is to ensure consistency of Pipeline behaviour with standalone use.")

        missing = set(self._limits) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")

        if (y is None) and self.drop_when_training:
            raise RuntimeError(
                "AnomalyCleaner was passed only X, when 'drop_when_training' was set to True "
                "This is not allowed, if the User was Training, pass y to ensure data consistency. "
                "If the User was Testing/Predicting, Then use AnomalyCleaner.transform(). "
                "If AnomalyCleaner was used in an imblearn.pipeline.Pipeline, explictly set 'drop_when_training' "
                "to False to mimic calling transform(), which will only cap. "
                "This error is a fail-safe to prevent from failed fit_resample() calls by the Pipeline, "
                "which is the default behaviour; and to ensure use of transform() instead, which is the intended behaviour."
            )
            
        elif (y is None) and (not self.drop_when_training):
            raise ValueError(
                "y was not passed to AnomalyCleaner.fit_resample() at the time of Training "
                "y must always be passed at the time of training, to be in accordance with the imbalanced-learn API"
            )

        if self.copy: X = X.copy(); y = y.copy()

        if self.drop_when_training:
            invalid = np.zeros(len(X), dtype=bool)
            
            for column, (low, high) in self._limits.items():
    
                invalid |= (
                    X[column].notna()
                    &
                    ~X[column].between(low, high)
                )
            self._trained = True
            return X.loc[~invalid], y.loc[~invalid]
        
        self._trained = True
        return self._capper.transform(X), y

    def transform(self, X, issue_warning=True):
        check_is_fitted(self,["_limits","_capper","_trained"],
                        msg="Sampler x Transformer, %(name)s must be trained before resampling. Please run fit_resample()")

        missing = set(self._limits) - set(X.columns)

        if missing: raise ValueError(f"{self.__class__.__name__} requires these columns, but they are missing: {sorted(missing)}")

        if (issue_warning) and (not self.grid_mode):
            warn(
                "AnomalyCleaner.transform() is meant only for testing/prediction purposes. "
                "It will strictly only cap the data. Useful if the User wishes to sample the data when training "
                "but transform at the time of testing/prediction. Which is the intended behaviour. "
                "It is interchangeable with fit_resample() if 'drop_when_training' is False. "
                "set 'issue_warning' explicitly to False in transform() if you wish to turn this warning off.",
                UserWarning
            )
        
        if self.copy: X = X.copy()

        return self._capper.transform(X)

    def fit_transform(self, X, y=None):
        raise NotImplementedError("This Method is specifically turned off for AnomalyCleaner, please use fit() and transform() seperately")

class DateTimeFeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    DOCSTRING

    This Transformer is to convert DateTimeFeatures into a more useful form.
    New columns added are

    PARAMETERS:
        > keep_end_features = Whether to keep accident End related features
        > mark_invalid_dates = Whether to mark invalid dates or not
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
                mark_invalid_dates = True,
                max_resolution_days = 1,
                copy = False):
        if max_resolution_days < 0: raise ValueError("max_resolution_days must be non-negative.")
        
        self.keep_end_features = keep_end_features
        self.mark_invalid_dates = mark_invalid_dates
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

        if self.mark_invalid_dates: 
            valid = self._validate_dates(X)
            
            X["Invalid Date Time Data"] = (~valid).astype("int8")
            
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

class OutOfCoreNumericalImputer(BaseEstimator, TransformerMixin):
    """
    DOCSTRING

    This Transformer has been designed to be Stateless and work with chunk based data.
    The Transformer needs to fit the Means at once and unincrementally to ensure that
    non-represetative means are not used for imputation of the data.

    PARAMETERS:
        > file_name = Path of the Data
        > columns = Columns to Mean-Impute
        > batch_size = The Batch Size for the chunks, it doesn't affect the final results
        > copy = Whether to return a copy of the dataframe or perform changes in-place
    """
    def __init__(self,*,
                 file_name,
                 columns,
                 batch_size = None,
                 copy = False):
        self.file_name = file_name
        self.columns = columns
        self.copy = copy
        self.batch_size = batch_size
    
    def fit(self, X=None, y=None):
        if self.batch_size is None: self.batch_size = 20_000
        
        means = {}
        for col in self.columns:
            _sum = 0
            _count = 0
            for chunk in pd.read_csv(self.file_name,index_col='ID',usecols=['ID',col],chunksize=self.batch_size):
                _sum += chunk[col].sum(skipna=True)
                _count += chunk.notna().sum()
            means[col] = _sum/_count

        self._means = means
        
        return self
        
    def transform(self, chunk):
        check_is_fitted(self, "_means")

        if sorted((chunk.columns).to_list()) != sorted(self.columns): raise ValueError('Transformation columns do not match fitted columns')
        
        if self.copy: chunk = chunk.copy()

        for col in self.columns:
            chunk[col] = chunk[col].fillna(self._means[col])
        
        return chunk