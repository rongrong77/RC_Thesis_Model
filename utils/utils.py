import numpy as np


def get_activity_index_test(test_labels, activity):
    activity_column_index = 1
    activity_index = [i for i, label in enumerate(test_labels) if label[activity_column_index] == activity]
    
    return activity_index

def get_exact_activity_index_test(test_labels, exact_activity):
    exact_activity_column = 1
    exact_activity_index = []
    for i, label in enumerate(test_labels):
        if np.all(label[:, exact_activity_column] == exact_activity):
            exact_activity_index.append(i)
    return exact_activity_index


def get_subject_index_test(test_labels, subject):
    subject_index = []
    subject_column = int(np.where(np.isin(test_labels[0][0], ["Subject01", "Subject02", "Subject03", "Subject04", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08", "Subject09", "Subject11", "Subject12","Subject13", "Subject14"],) == True)[0])
    for i, label in enumerate(test_labels):
        if np.all(label[:, subject_column] == subject):
            subject_index.append(i)
    return subject_index


def get_model_name_from_activites(train_activity, test_activity):
    original_name = {"levelwalking": "g1", "ramp": "r", "stair": "s", "obstacle": "o"}
    model_train_activity = [original_name[activity] for activity in train_activity]
    model_test_activity = [original_name[activity] for activity in test_activity]
    return model_train_activity, model_test_activity