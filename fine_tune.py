# Example to fine-tune on 3000 samples from Cifar10

from load_cifar10 import load_cifar10_data
from models.densenet161 import densenet161_model

img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 10
batch_size = 8
nb_epoch = 10


print("loading data")
# Load Cifar10 data. Please implement your own load_data() module for your own dataset
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

# Load our model
model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

# Start Fine-tuning
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

from sklearn.metrics import log_loss

# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)