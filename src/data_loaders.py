from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_loaders(train_dir, test_dir, img_size, batch_size):
    train_gen = ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    # val_ds = train_gen.flow_from_directory(
    #     val_dir,
    #     target_size=(img_size, img_size),
    #     batch_size=batch_size,
    #     shuffle=False
    # )
    test_ds = train_gen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )
    return train_ds, test_ds

# if __name__ == '__main__':
#     train_dir = 'data/train'
#     test_dir = 'data/test'
#     img_size = 128
#     batch_size = 32

#     train_ds, test_ds = get_data_loaders(train_dir, test_dir, img_size, batch_size)

#     images, labels = next(train_ds)
