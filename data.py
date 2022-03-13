from torch.utils.data import Dataset
import torch

import numpy as np

def get_class_i(x, y, class_list):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    X_sub = []
    y_sub = []
    for i in class_list:
        # Convert to a numpy array
        y = np.array(y)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(y == i)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        x_i = [x[j] for j in pos_i]
        y_i = y[pos_i].tolist()
        X_sub.extend(x_i)
        y_sub.extend(y_i)

    return np.asarray(X_sub), np.asarray(y_sub)

class CIFARCustom(Dataset):
    def __init__(self, X, y, class_dict, classes) -> None:
        super(CIFARCustom, self).__init__()
        self.X, self.y = get_class_i(X, y, [class_dict[i] for i in classes])
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return torch.as_tensor(self.X[i].T), torch.as_tensor(self.y[i])


