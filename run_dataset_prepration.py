from utils.read_write import read_csv
import os
import pickle
from config import get_config_universal

dataset_path = './Datasets/'

class DataReader:
    def __init__(self, config, subject_list, activity_list):
        self.config = config
        self.subject_list = subject_list
        self.activity_list = activity_list

    def read_data(self, data_type):
        data = {}
        for subject in self.subject_list:
            if subject not in data:
               data[subject] = {}
            subject_dir_temp = config['dataset_path'] + subject + '/'
            subject_dir = subject_dir_temp + os.listdir(subject_dir_temp)[0]
            for activity in activity_list:
                if activity not in data[subject]:
                    data[subject][activity] = []
                activity_dir = os.path.join(subject_dir, activity, data_type) 
                for (root, dirs, files) in os.walk(subject_dir + '/' + activity + '/' + data_type + '/'):
                    for name in files:
                        if name[-4:] == '.csv':
                            file = os.path.join(root, name)
                            df = read_csv(file)
                            data[subject][activity].append(df)
        return data


config = get_config_universal('oemal')
activity_list = ['levelwalking', 'ra', 'sa', 'rd', 'sd']
datatype_list = ['imu', 'ia']
subject_list = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08", "Subject09", "Subject10","Subject11", "Subject12",
"Subject13", "Subject14"]


datareader_handler = DataReader(config, subject_list, activity_list)
dl_dataset = './Datasets/MiniDataset/'
dataset_file = dl_dataset + "".join(activity_list) + '_' + "".join(datatype_list) + '_' + "".join(subject_list) + '.p'
dataset = {}
dataset['dataset_info'] = [('dataset_name', 'oemal'),
                           ('activity_list', activity_list),
                           ('subject_list', subject_list),
                           ('datatype_list', datatype_list)]

for datatype in datatype_list:
    data = datareader_handler.read_data(datatype)
    dataset[datatype] = data

if os.path.isfile(dataset_file):
    print('file exist')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
else:
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


print("The highest pickle protocol available is:", pickle.HIGHEST_PROTOCOL)


