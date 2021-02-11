import numpy as np
from pymining import itemmining

from src.predictive_model.predictive_model import drop_columns


def compute_feedback(explanations, predictive_model, test_df, encoder, top_k=None):
    predicted = predictive_model.model.predict(drop_columns(test_df))
    actual = test_df['label']

    trace_ids = test_df['trace_id']

    confusion_matrix = _retrieve_confusion_matrix_ids(trace_ids, predicted, actual, encoder)

    filtered_explanations = _filter_explanations(explanations, threshold=13)

    frequent_patterns = _mine_frequent_patterns(confusion_matrix, filtered_explanations)

    feedback = {
        classes: _subtract_patterns(
            sum([frequent_patterns[classes][cl] for cl in confusion_matrix.keys()], []),
            frequent_patterns[classes][classes]
        )
        for classes in confusion_matrix.keys()
    }

    if top_k is not None:
        for classes in feedback:
            feedback[classes] = feedback[classes][:top_k]

    return feedback


def _retrieve_confusion_matrix_ids(trace_ids, actual, predicted, encoder) -> dict:
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
    for element in confusion_matrix:
        print(element, [  # print importances for debug
            [
                str(feature_name) + ' // ' + str(value) + ' // ' + str(importance)
                for feature_name, value, importance in filtered_explanations[tid]
            ]
            for tid in confusion_matrix[element]
        ])
        mined_patterns[element] = itemmining.relim(itemmining.get_relim_input([
            [
                str(feature_name) + '//' + str(value)  # + '_' + str(_tassellate_number(importance))
                for feature_name, value, importance in filtered_explanations[tid]
            ]
            for tid in confusion_matrix[element]
        ]), min_support=2)
        mined_patterns[element] = sorted(
            [
                ([el.split('//') for el in list(key)], mined_patterns[element][key])
                for key in mined_patterns[element]
            ],
            key=lambda x: x[1],
            reverse=True
        )

    return mined_patterns


def _tassellate_number(element):
    element = str(element).split('.')
    return element[0] + '.' + element[1][:3]


def _subtract_patterns(list1, list2):

    difference = [el[0] for el in list1]
    for el, _ in list2:
        if el in difference:
            difference.remove(el)

    return difference

