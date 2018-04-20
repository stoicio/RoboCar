import argparse
from keras.callbacks import ModelCheckpoint
import os
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from navigator.utils import dataset, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Steering Model')
    parser.add_argument(
        'log',
        type=str,
        help='Path to driving log CSV file'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default='./data/',
        help='Path to image folder, where the images are stored'
    )
    args = parser.parse_args()

    driving_log = args.log
    base_data_dir = args.data_dir

    if not os.path.exists(driving_log) or not os.path.exists(base_data_dir):
        raise ValueError('Driving Log or base Dir does not exist')
    
    df = pandas.read_csv(driving_log)
    image_path_resolver = lambda x: os.path.join(base_data_dir, x['center'].strip())  # flake8: noqa
    df['center'] = df.apply(image_path_resolver, axis=1)

    all_x, all_y = shuffle(df.center.values, df.steering.values)
    # print(all_x[0])
    X_train, X_val, y_train, y_val = train_test_split(all_x, all_y, test_size=0.2)

    
    save_all_epochs = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                      monitor='val_loss', save_best_only=False, )

    model, input_shape = model.get_model()

    training_data = dataset.batch_generator(X_train, y_train, input_shape)
    validation_data = dataset.batch_generator(X_val, y_val, input_shape, training=False)
    
    model.fit_generator(training_data, steps_per_epoch=50, epochs=5, verbose=2,
                        validation_data=validation_data, validation_steps=12,
                        callbacks= [save_all_epochs])
