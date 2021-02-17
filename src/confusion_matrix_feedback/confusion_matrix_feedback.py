import numpy as np
from pymining import itemmining

from src.encoding.data_encoder import PADDING_VALUE
from src.predictive_model.predictive_model import drop_columns


def compute_feedback(explanations, predictive_model, feedback_df, encoder, threshold=None, top_k=None):
    predicted = predictive_model.model.predict(drop_columns(feedback_df))
    actual = feedback_df['label']

    trace_ids = feedback_df['trace_id']

    confusion_matrix = _retrieve_confusion_matrix_ids(trace_ids, predicted=predicted, actual=actual, encoder=encoder)

    filtered_explanations = _filter_explanations(explanations, threshold)

    frequent_patterns = _mine_frequent_patterns(confusion_matrix, filtered_explanations)

    feedback = {
        classes: _subtract_patterns(
            sum([frequent_patterns[classes][cl] for cl in confusion_matrix.keys()], []),
            frequent_patterns[classes][classes]
        )
        for classes in confusion_matrix.keys()
    }

    feedback = {
        classes: sorted(feedback[classes], key=lambda x: x[1] * len(x[0]) / len(feedback[classes]), reverse=True)
        for classes in feedback
    }

    if top_k is not None:
        for classes in feedback:
            feedback[classes] = feedback[classes][:top_k]

    return feedback


def _retrieve_confusion_matrix_ids(trace_ids, predicted, actual, encoder) -> dict:
    decoded_predicted = encoder.decode_column(predicted, 'label')
    decoded_actual = encoder.decode_column(actual, 'label')
    elements = np.column_stack((
        trace_ids,
        decoded_predicted,
        decoded_actual
    )).tolist()

    # matrix format is (actual, predicted)
    confusion_matrix = {}
    classes = list(encoder.get_values('label')[0])
    if str(PADDING_VALUE) in classes: classes.remove(str(PADDING_VALUE))
    for act in classes:
        confusion_matrix[act] = {}
        for pred in classes:
            confusion_matrix[act][pred] = {
                trace_id
                for trace_id, predicted, actual in elements
                if actual == act and predicted == pred
            }

    return confusion_matrix


def _filter_explanations(explanations, threshold=None):
    if threshold is None:
        threshold = min(13, int(max(len(explanations[tid]) for tid in explanations) * 10 / 100) + 1)
    return {
        trace_id:
            sorted(explanations[trace_id], key=lambda x: x[2], reverse=True)[:threshold]
        for trace_id in explanations
    }


def _mine_frequent_patterns(confusion_matrix, filtered_explanations):
    mined_patterns = {}
    for actual in confusion_matrix:
        mined_patterns[actual] = {}
        for pred in confusion_matrix[actual]:
            mined_patterns[actual][pred] = itemmining.relim(itemmining.get_relim_input([
                [
                    str(feature_name) + '//' + str(value)  # + '_' + str(_tassellate_number(importance))
                    for feature_name, value, importance in filtered_explanations[tid]
                ]
                for tid in confusion_matrix[actual][pred]
                if tid in filtered_explanations
            ]), min_support=2)
            mined_patterns[actual][pred] = sorted(
                [
                    ([el.split('//') for el in list(key)], mined_patterns[actual][pred][key])
                    for key in mined_patterns[actual][pred]
                ],
                key=lambda x: x[1],
                reverse=True
            )

    return mined_patterns


def _tassellate_number(element):
    element = str(element).split('.')
    return element[0] + '.' + element[1][:3]


def _subtract_patterns(list1, list2):

    difference = list1
    for el, _ in list2:
        if el in [e[0] for e in difference]:
            index = [e[0] for e in difference].index(el)
            difference.remove(difference[index])

    return difference

