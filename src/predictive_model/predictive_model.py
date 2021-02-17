import logging

from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.common import evaluate
from src.predictive_model.common import PredictionMethods

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'], 1)
    return df


class PredictiveModel:

    def __init__(self, model_type, train_df, validate_df):
        self.model_type = model_type
        self.config = None
        self.model = None
        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)

    def train_and_evaluate_configuration(self, config, target):
        try:
            model = self._instantiate_model(config)

            model.fit(self.train_df, self.full_train_df['label'])

            predicted = model.predict(self.validate_df)
            scores = model.predict_proba(self.validate_df)[:, 1]

            actual = self.full_validate_df['label']

            result = evaluate(actual, predicted, scores, loss=target)

            return {
                'status': STATUS_OK,
                'loss': - result['loss'], #we are using fmin for hyperopt
                'exception': None,
                'config': config,
                'model': model,
                'result': result,
            }
        except Exception as e:
            return {
                'status': STATUS_FAIL,
                'loss': 0,
                'exception': str(e)
            }

    def _instantiate_model(self, config):
        if self.model_type == PredictionMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)
        elif self.model_type == PredictionMethods.LSTM.value:
            raise Exception('not yet supported model_type')
        else:
            raise Exception('unsupported model_type')
        return model

