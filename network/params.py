from model import get_model_vgg16_pretrained
from model import get_model_vgg19_pretrained
from model import get_model_resnet50_pretrained # (197, 197, 3)
from model import get_model_mobilenet_pretrained # (128, 128, 3)
from model import get_model_inceptionv3_pretrained # (139, 139, 3)

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
