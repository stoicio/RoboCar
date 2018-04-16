import h5py
import numpy as np


def write_dataset(data, labels, path, dataset_name, write_method='w'):
    '''
    Create a HDF5 dataset with the given options
    write_method (char) - specifies the mode in which to open the dataset in.
                          Standard file flags apply
                          default('x') - throws an error if file already exists
    '''
    with h5py.File(path, write_method) as db:
        shape = (len(data), len(data[0]) + 1)  # Shape (Num_Samples, Num_Features + 1) +1 for labels
        dataset = db.create_dataset(dataset_name, shape, dtype='float')
        dataset[0:len(data)] = np.c_[labels, data]


def load_dataset(path, dataset_name):
    '''
    Loads data from  HDF5 database in the given path
    '''
    with h5py.File(path, 'r') as db:
        # Load the saved dataset
        (labels, data) = (db[dataset_name][:, 0], db[dataset_name][:, 1:])
    return (data, labels)
