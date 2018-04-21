import argparse
from time import gmtime, strftime
from keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
import os
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from navigator.utils import dataset, models

SAMPLES_PER_EPOCH = 20000
BATCH_SIZE = 128
BATCHES_PER_EPOCH = SAMPLES_PER_EPOCH // BATCH_SIZE
NUM_EPOCHS = 20
LEARNING_RATE = 0.005
DROPOUT_RATE = 0.4

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def get_save_path(base_data_dir):
    folder_name = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
    folder_path = os.path.join(base_data_dir, folder_name)
    os.makedirs(folder_path)
    logger.info('Saving models to %s', folder_path)
    return os.path.join(folder_path, 'model.{epoch:02d}-{val_loss:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Steering Model')
    parser.add_argument('-l', '--log', type=str, default='./data/driving_log.csv',
                        help='Path to driving log CSV file')

    parser.add_argument('-m', '--model', type=str, default='nvidia',
                        help='Model to use', choices= ['nvidia', 'comma.ai'])

    parser.add_argument('-d', '--data_dir', type=str, default='./data/',
                        help='Path to image folder, where the images are stored')


    args = parser.parse_args()

    driving_log = args.log
    base_data_dir = args.data_dir
    model_name = args.model

    if not os.path.exists(driving_log) or not os.path.exists(base_data_dir):
        raise ValueError('Driving Log or base Dir does not exist')
    
    # Load Image Path from CSV
    df = pandas.read_csv(driving_log)
    # image_path_resolver = lambda x: os.path.join(base_data_dir, x['center'].strip())  # flake8: noqa
    # df['center'] = df.apply(image_path_resolver, axis=1)

    all_x, all_y = shuffle(df.center.values, df.steering.values)

    X_train, X_val, y_train, y_val = train_test_split(all_x, all_y, test_size=0.2)

    # Make validation steps divisible by batch size so that there are no repeated samples
    num_validation_steps = (len(y_val) - (len(y_val) % BATCH_SIZE)) / BATCH_SIZE
    model_save_paths = get_save_path(base_data_dir)
    save_all_epochs = ModelCheckpoint(model_save_paths, monitor='val_loss',
                                      save_best_only=False, )
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0.003, patience=2, verbose=2, mode='min')

    if model_name == 'nvidia':
        model, input_shape = models.get_nvidia_model(dropout_prob=DROPOUT_RATE,
                                                     learning_rate=LEARNING_RATE)

    training_data = dataset.batch_generator(X_train, y_train, input_shape,
                                            zero_angle_retention_rate=0.05)
    validation_data = dataset.batch_generator(X_val, y_val, input_shape, training=False)
    
    logger.info('          Training Parameters         ')
    logger.info('======================================')
    logger.info('Batch size        : %d', BATCH_SIZE)
    logger.info('Samples per epoch : %d', SAMPLES_PER_EPOCH)
    logger.info('Training b/epoch  : %d', BATCHES_PER_EPOCH)
    logger.info('Validation b/epoch: %d', num_validation_steps)
    logger.info('Total Epochs      : %d', NUM_EPOCHS)
    logger.info('Learning rate     : %f', LEARNING_RATE)
    logger.info('Dropout rate      : %f', DROPOUT_RATE)
    
    model.fit_generator(training_data, steps_per_epoch=BATCHES_PER_EPOCH, 
                        epochs=NUM_EPOCHS, verbose=2, validation_data=validation_data,
                        validation_steps=num_validation_steps, callbacks= [save_all_epochs,stop_early])
