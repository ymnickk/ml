import pickle
from keras.models import load_model


def get_models(numbers_model: list = None):
    models_list = []
    return_list = []

    with open('models/logreg.pkl', 'rb') as pickle_in:
        model_1 = pickle.load(pickle_in)
        models_list.append(model_1)
    with open('models/kmeans.pkl', 'rb') as pickle_in:
        model_2 = pickle.load(pickle_in)
        models_list.append(model_2)
    with open('models/gradboost.pkl', 'rb') as pickle_in:
        model_3 = pickle.load(pickle_in)
        models_list.append(model_3)
    with open('models/bagging.pkl', 'rb') as pickle_in:
        model_4 = pickle.load(pickle_in)
        models_list.append(model_4)
    with open('models/stacking.pkl', 'rb') as pickle_in:
        model_5 = pickle.load(pickle_in)
        models_list.append(model_5)

    model_6 = load_model("models/my_model.h5")
    models_list.append(model_6)

    if not numbers_model:
        return model_1, model_2, model_3, model_4, model_5, model_6

    for number_model in numbers_model:
        return_list.append(models_list[number_model - 1])

    return tuple(return_list)
