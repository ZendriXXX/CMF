import logging

from src.encoding.common import get_encoded_df, EncodingType
from src.evaluation.common import evaluate
from src.explanation.common import explain, ExplainerType
from src.confusion_matrix_feedback.confusion_matrix_feedback import compute_feedback
from src.confusion_matrix_feedback.randomise_features import randomise_features
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import PredictionMethods
from src.predictive_model.predictive_model import PredictiveModel, drop_columns

logger = logging.getLogger(__name__)

logger.info('Hey there!')

OFFSET_DATA = 'input_data/'
TRAIN_DATA = OFFSET_DATA + 'd1_train_explainability_0-38.xes'
VALIDATE_DATA = OFFSET_DATA + 'd1_validation_explainability_38-40.xes'
TEST_DATA = OFFSET_DATA + 'd1_test_explainability_40-50.xes'
RETRAIN_TEST_DATA = OFFSET_DATA + 'd1_test2_explainability_50-60.xes'

OUTPUT_DATA = 'output_data'

# global CONF
CONF = { #This contains the configuration for the run
    'prefix_length': 5,
    'padding': True,
    'feature_selection': EncodingType.SIMPLE.value,
    'labeling_type': LabelTypes.NEXT_ACTIVITY.value,
    'predictive_model': PredictionMethods.RANDOM_FOREST.value,
    'explanator': ExplainerType.SHAP.value,
    'hyperparameter_optimisation': True,
    'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
    'hyperparameter_optimisation_epochs': 3
}

logger.debug('LOAD DATA')
train_log = get_log(filepath=TRAIN_DATA)
validate_log = get_log(filepath=VALIDATE_DATA)
test_log = get_log(filepath=TEST_DATA)
retrain_test_log = get_log(filepath=RETRAIN_TEST_DATA)

logger.debug('ENCODE DATA')
encoder, train_df, validate_df, test_df, retrain_test_df = get_encoded_df(
    train_log=train_log,
    validate_log=validate_log,
    test_log=test_log,
    retrain_test_log=retrain_test_log,
    CONF=CONF
)

logger.debug('TRAIN PREDICTIVE MODEL')
predictive_model = PredictiveModel(CONF['predictive_model'], train_df, validate_df)

predictive_model.model, predictive_model.config = retrieve_best_model(
    predictive_model,
    CONF['predictive_model'],
    max_evaluations=CONF['hyperparameter_optimisation_epochs'],
    target=CONF['hyperparameter_optimisation_target']
)

logger.debug('EVALUATE PREDICTIVE MODEL')
predicted = predictive_model.model.predict(drop_columns(test_df))
scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
actual = test_df['label']
initial_result = evaluate(actual, predicted, scores)

logger.debug('COMPUTE EXPLANATION')
explanations = explain(CONF['explanator'], predictive_model, test_df, encoder)

logger.debug('COMPUTE FEEDBACK')
feedback = compute_feedback(explanations, predictive_model, test_df, encoder, top_k=1)

logger.debug('SHUFFLE FEATURES')
encoder.decode(train_df)
shuffled_train_df = randomise_features(feedback, train_df)
encoder.encode(shuffled_train_df)

logger.debug('RETRAIN-- TRAIN PREDICTIVE MODEL')
predictive_model = PredictiveModel(CONF['predictive_model'], shuffled_train_df, validate_df)

predictive_model.model, predictive_model.config = retrieve_best_model(
    predictive_model,
    CONF['predictive_model'],
    max_evaluations=CONF['hyperparameter_optimisation_epochs'],
    target=CONF['hyperparameter_optimisation_target']
)

logger.debug('RETRAIN-- EVALUATE PREDICTIVE MODEL')
predicted = predictive_model.model.predict(drop_columns(retrain_test_df))
scores = predictive_model.model.predict_proba(drop_columns(retrain_test_df))[:, 1]
actual = retrain_test_df['label']
retrain_result = evaluate(actual, predicted, scores)

logger.info('RESULT')
logger.info('INITIAL', initial_result)
logger.info('RETRAIN', retrain_result)

logger.info('Done, cheers!')
