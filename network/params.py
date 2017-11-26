from model import get_model_sequential, get_model_residual

seed = 13
max_epochs = 1000
batch_size = 64
model_factory = get_model_sequential
validation_split = 0.15
best_weights_path = 'weights/best_weights.hdf5'
best_model_path = 'models/best_model.json'
tta_steps = 10
num_folds = 8
