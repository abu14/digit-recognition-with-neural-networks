from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from .model import create_model  # Import from the model module

def train_model(model, x_train, y_train, x_val, y_val, epochs=40, batch_size=10):  # Added parameters
    """Trains the CNN model with data augmentation."""

    datagen = ImageDataGenerator(zoom_range=0.1, height_shift_range=0.1,
                                width_shift_range=0.1, rotation_range=10)

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x) #Start learning rate at 0.001

    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) // batch_size, # Calculate steps per epoch dynamically
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_val, y_val),
                        callbacks=[annealer])
    return history