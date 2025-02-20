import itertools
import logging
from statistics import stdev, mean

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

def dict_mean(dict_list):
    mean_dict = {
        'avg': dict(),
        'min': dict(),
        'max': dict(),
        'stdev': dict()
    }
    for key in dict_list[0].keys():
        values = [d[key] for d in dict_list]
        mean_dict['avg'][key] = mean(values)
        mean_dict['min'][key] = min(values)
        mean_dict['max'][key] = max(values)
        mean_dict['stdev'][key] = stdev(values)
    return mean_dict


def run_full_pipeline(CONF=None):
    logger.info('Hey there!')
    if CONF is None:
        CONF = {  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': 'input_data/' + 'd1_train_explainability_0-38.xes',
                    'VALIDATE_DATA': 'input_data/' + 'd1_validation_explainability_38-40.xes',
                    'FEEDBACK_DATA': 'input_data/' + 'd1_test_explainability_40-50.xes',
                    'TEST_DATA': 'input_data/' + 'd1_test2_explainability_50-60.xes',
                    'OUTPUT_DATA': 'output_data',
                },
            'prefix_length': 5,
            'padding': True,
            'feature_selection': EncodingType.SIMPLE.value,
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': PredictionMethods.RANDOM_FOREST.value,
            'explanator': ExplainerType.SHAP.value,
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': True,
            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
            'hyperparameter_optimisation_epochs': 100
        }

    logger.debug('LOAD DATA')
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    feedback_log = get_log(filepath=CONF['data']['FEEDBACK_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    logger.debug('ENCODE DATA')
    encoder, train_df, validate_df, feedback_df, test_df = get_encoded_df(
        train_log=train_log,
        validate_log=validate_log,
        test_log=feedback_log,
        retrain_test_log=test_log,
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
    explanations = explain(CONF['explanator'], predictive_model, feedback_df, encoder)

    logger.debug('COMPUTE FEEDBACK')
    feedback_10 = compute_feedback(
        explanations,
        predictive_model,
        feedback_df,
        encoder,
        top_k=CONF['top_k']
    )

    logger.debug('SHUFFLE FEATURES')
    encoder.decode(train_df)
    encoder.decode(validate_df)
    returned_results = {}
    feedback = {}
    for top_k_threshold in [1, 2]:
        feedback[top_k_threshold] = {classes: feedback_10[classes][:top_k_threshold] for classes in feedback_10}

        retrain_results = []
        for _ in range(10):

            shuffled_train_df = randomise_features(feedback[top_k_threshold], train_df)
            shuffled_validate_df = randomise_features(feedback[top_k_threshold], validate_df)
            encoder.encode(shuffled_train_df)
            encoder.encode(shuffled_validate_df)

            logger.debug('RETRAIN-- TRAIN PREDICTIVE MODEL')
            predictive_model = PredictiveModel(CONF['predictive_model'], shuffled_train_df, shuffled_validate_df)
            try:
                predictive_model.model, predictive_model.config = retrieve_best_model(
                    predictive_model,
                    CONF['predictive_model'],
                    max_evaluations=CONF['hyperparameter_optimisation_epochs'],
                    target=CONF['hyperparameter_optimisation_target']
                )

                logger.debug('RETRAIN-- EVALUATE PREDICTIVE MODEL')
                predicted = predictive_model.model.predict(drop_columns(test_df))
                scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
                actual = test_df['label']
                retrain_results += [evaluate(actual, predicted, scores)]
            except Exception as e:
                pass

        stats_retrain_results = dict_mean(retrain_results)
        returned_results[top_k_threshold] = {
            'avg': stats_retrain_results['avg'],
            'min': stats_retrain_results['min'],
            'max': stats_retrain_results['max'],
            'stdev': stats_retrain_results['stdev'],
            'retrain_results': retrain_results
        }

    logger.info('RESULT')
    logger.info('INITIAL', initial_result)
    logger.info('RETRAIN', returned_results)

    logger.info('Done, cheers!')

    return {'feedback_10': feedback_10, 'used_feedback': feedback, 'initial_result': initial_result, 'retrain_result': returned_results}


if __name__ == '__main__':
    run_full_pipeline()

