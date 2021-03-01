from lime import lime_tabular
import numpy as np

from src.predictive_model.predictive_model import drop_columns


def lime_explain(predictive_model, full_test_df, encoder):
    test_df = drop_columns(full_test_df)

    labels = list(set(full_test_df['label']))

    explainer = _init_explainer(test_df, labels)
    importances = _get_explanation(explainer, full_test_df, encoder, predictive_model.model, labels)

    return importances


def _init_explainer(df, labels):
    return lime_tabular.LimeTabularExplainer(
        df.as_matrix(),
        mode='classification',
        training_labels=labels,
        feature_names=df.columns,
        categorical_features=[
            i
            for i in range(len(df.columns))
            # if 'age' not in df.columns[i].lower() and \
            #    'time' not in df.columns[i].lower()
        ], #todo <- can we track cat noncat features?
        verbose=False,
        discretize_continuous=True
    )


def _get_explanation(explainer, target_df, encoder, model, labels):
    return {
        str(row['trace_id']):
            np.column_stack((
                target_df.columns[1:-1],
                encoder.decode_row(row)[1:-1],
                [
                    element[1]
                    for element in _explain_instance(explainer, row, model, target_df, labels)
                ]
            )).tolist()
        for _, row in target_df.iterrows()
    }


def _explain_instance(explainer, row, model, target_df, labels):
    retval = explainer.explain_instance(
        drop_columns(row.to_frame(0).T).tail(1).squeeze(),
        model.predict_proba,
        num_features=len(target_df.columns),
        labels=[ i - 1 for i in labels ]
    )
    predicted_value = int(model.predict(drop_columns(row.to_frame().T))[0]) - 1
    retval = retval.local_exp[predicted_value]
    retval = sorted(retval, key=lambda x: x[0])
    return retval

