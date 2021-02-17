from lime import lime_tabular
import numpy as np

from src.predictive_model.predictive_model import drop_columns


def lime_explain(predictive_model, full_test_df, encoder):
    test_df = drop_columns(full_test_df)

    explainer = _init_explainer(test_df)
    importances = _get_explanation(explainer, full_test_df, encoder, predictive_model.model)

    return importances


def _init_explainer(df):
    return lime_tabular.LimeTabularExplainer(
        df.as_matrix(),
        feature_names=df.columns,
        categorical_features=[i for i in range(len(df.columns))],
        verbose=True,
        mode='classification',
    )


def _get_explanation(explainer, target_df, encoder, model):
    return {
        str(row['trace_id']):
            np.column_stack((
                target_df.columns[1:-1],
                encoder.decode_row(row)[1:-1],
                [
                    element[1]
                    for element in _explain_instance(explainer, row, model, target_df).as_list()
                ]
            )).tolist()
        for _, row in target_df.iterrows()
    }


def _explain_instance(explainer, row, model, target_df):
    retval = explainer.explain_instance(
                    drop_columns(row.to_frame(0).T).tail(1).squeeze(),
                    model.predict_proba,
                    num_features=len(target_df.columns))
    return retval

