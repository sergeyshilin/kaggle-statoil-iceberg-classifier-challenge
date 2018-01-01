from model import get_model_vgg16_pretrained
from model import get_model_vgg16
from model import get_model_vgg19_pretrained
from model import get_model_resnet50_pretrained # (197, 197, 3)
from model import get_model_mobilenet_pretrained # (128, 128, 3)
from model import get_model_inceptionv3_pretrained # (139, 139, 3)
from model import get_model_xception_pretrained
from model import get_model_custom

from utils import identical
from utils import data_normalization
from utils import color_composition

seed = 13
max_epochs = 1000
batch_size = 128
model_factory = get_model_vgg16_pretrained

data_adapt = identical # data_normalization
pseudolabeling = True

model_input_size = (75, 75, 3)
best_weights_path = 'weights/best_weights.hdf5'
best_weights_checkpoint = 'weights/best_weights_checkpoint.hdf5'
best_model_path = 'models/best_model.json'
tta_steps = 10
num_folds = 8
learning_rates = [0.001 * 3., 0.001, 0.001 / 3.]

## Augmentation parameters
aug_horizontal_flip = True
aug_vertical_flip = True
aug_rotation = 10
aug_width_shift = 0.05
aug_height_shift = 0.05
aug_channel_shift = 0.0
aug_shear = 0.0
aug_zoom = 0.2
