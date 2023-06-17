from src.class_names import get_class_names
from src.data_loaders import get_data_loaders
from src.utils import counter
from src.cnn_model import create_cnn_model

train_dir = 'data/train'
test_dir = 'data/test'

class_names, num_classes = get_class_names(train_dir)
print(class_names)
print(f'Number of Classes: {num_classes}')

img_size = 128
batch_size = 32


train_ds,  test_ds = get_data_loaders(train_dir, test_dir, img_size, batch_size)

images, labels = next(train_ds)

counter(train_dir)

input_shape = (img_size, img_size, 3)

model = create_cnn_model(input_shape, num_classes)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=10,
                    validation_data = test_ds,
                    )

model.save('models/cnnmodel.h5')


