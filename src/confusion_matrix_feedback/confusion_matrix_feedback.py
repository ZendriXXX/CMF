import numpy as np
from pymining import itemmining

from src.predictive_model.predictive_model import drop_columns


def compute_feedback(explanations, predictive_model, test_df, encoder, top_k=None):
    predicted = predictive_model.model.predict(drop_columns(test_df))
    actual = test_df['label']

    trace_ids = test_df['trace_id']

    confusion_matrix = _retrieve_confusion_matrix_ids(trace_ids, predicted, actual, encoder)

    filtered_explanations = _filter_explanations(explanations, confusion_matrix)

    frequent_patterns = _mine_frequent_patterns(confusion_matrix, filtered_explanations)

    feedback = {
        'true': _subtract_patterns(frequent_patterns['fp'], frequent_patterns['tp']),
        'false': _subtract_patterns(frequent_patterns['fn'], frequent_patterns['tn'])
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

    tp = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'true' and predicted == 'true'
    }
    tn = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'false' and predicted == 'false'
    }
    fp = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'false' and predicted == 'true'
    }
    fn = {
        trace_id
        for trace_id, predicted, actual in elements
        if actual == 'true' and predicted == 'false'
    }
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def _filter_explanations(explanations, confusion_matrix, thresholds=None):
    true = list(confusion_matrix['tp']) + list(confusion_matrix['fp'])
    false = list(confusion_matrix['tn']) + list(confusion_matrix['fn'])

    if thresholds is None:
        thresholds = {
            'true': np.mean([ feature[2] for tid in true for feature in explanations[tid] if feature[2] > 0 ]),
            'false': np.mean([ feature[2] for tid in false for feature in explanations[tid] if feature[2] < 0 ])
        }

    filtered_explanations = {}
    for trace_id in explanations:
        filtered_explanations[trace_id] = []
        for feature_name, value, importance in explanations[trace_id]:
            if trace_id in true and importance > thresholds['true']:  # this becomes a for each in multiclass
                filtered_explanations[trace_id] += [ (feature_name, value, importance) ]
            elif trace_id in false and importance < thresholds['false']:
                filtered_explanations[trace_id] += [ (feature_name, value, importance) ]

    return filtered_explanations


def _mine_frequent_patterns(confusion_matrix, filtered_explanations):
    mined_patterns = {}
    for element in confusion_matrix:
        mined_patterns[element] = itemmining.relim(itemmining.get_relim_input([
            [
                str(feature_name) + '//' + str(value) # + '_' + str(_tassellate_number(importance))
                for feature_name, value, importance in filtered_explanations[tid]
            ]
            for tid in confusion_matrix[element]
        ]), min_support=2)
        mined_patterns[element] = sorted(
            [
                ([ el.split('//') for el in list(key) ], mined_patterns[element][key])
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

