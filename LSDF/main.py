# -*- coding: utf-8 -*-
# @Time    : 2023/12/31
# @Author  : Yang Li


from sklearn import model_selection
from sklearn.model_selection import train_test_split

from model.LSDFmodel import LSDFModel
from utils.myutils import compute_metrics, load, ResultHandling


def experiment(para_path, para_filename):
    # load dataset
    prop, data, target = load(para_path, para_filename)  # n × m and n × q

    # cross validation
    metric_score = []
    cross = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in cross.split(data, target):
        # prepare dataset
        train_data, val_data, train_target, val_target = train_test_split(data[train], target[train], test_size=0.2,
                                                                          random_state=42)
        train = train_data, train_target
        val = val_data, val_target
        test = data[test], target[test]

        # train
        model = LSDFModel(prop)
        model.train(train, val)

        # predict
        predict = model.predict(test[0])
        metric_score.append(compute_metrics(predict, test[1]))

    # show and save
    rh = ResultHandling(para_filename, metric_score)
    rh.show()
    rh.save_results()


if __name__ == "__main__":
    path = "../Datasets"
    datasets = ["Birds", "Business", "Cal500", "CHD49", "Computer", "Emotion", "Enron", "Entertainment", "Flags",
                "Foodtruck", "Genbase", "Image", "Scene", "Society", "WaterQuality", "Yeast"]

    for dataset in datasets:
        experiment(path, dataset)
