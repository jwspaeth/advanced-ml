class hw2_dataset():
    
    data_path = "/home/fagg/ml_datasets/bmi/bmi_dataset.pkl"

    def __init__(self):

        with open("/home/fagg/ml_datasets/bmi/bmi_dataset.pkl", "rb") as fp:
            self.data = pickle.load(fp)

    def load_fold_rotation(rotation_index):
        pass

    def get_n_folds(self):
        return 20