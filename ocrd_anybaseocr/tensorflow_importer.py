__all__ = ['tf']

import os
import warnings
from ocrd_utils import initLogging, getLogger
initLogging()
getLogger('tensorflow').setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # No prints from the tensorflow side
warnings.filterwarnings('ignore', category=FutureWarning)
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
tf.disable_v2_behavior()
