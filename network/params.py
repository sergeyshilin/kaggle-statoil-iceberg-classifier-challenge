from model import get_model_vgg16_pretrained
from model import get_model_vgg19_pretrained
from model import get_model_resnet
from model import get_model_mobilenet

seed = 13
max_epochs = 1000
batch_size = 64
model_factory = get_model_vgg16_pretrained
model_input_size = (75, 75, 3)
validation_split = 0.15
best_weights_path = 'weights/best_weights.hdf5'
best_model_path = 'models/best_model.json'
tta_steps = 10
num_folds = 8
