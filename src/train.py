from class_names import get_class_names
from data_loaders import get_data_loaders
from utils import counter
from cnn_model import create_cnn_model

train_dir = 'data/train'
test_dir = 'data/test'

class_names, num_classes = get_class_names(train_dir)
print(class_names)
print(f'Number of Classes: {num_classes}')


img_size = 224
batch_size = 32

train_ds,  test_ds = get_data_loaders(train_dir, test_dir, img_size, batch_size)

images, labels = next(train_ds)

counter(train_dir)

input_shape = (img_size, img_size, 3)

model = create_cnn_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
                    epochs=30,
                    validation_data = test_ds
                    )