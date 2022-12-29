import tensorflow_hub as hub
import tensorflow as tf
from preprocess_functions import *


model_path = "skimlit_model_200k"
txt_path = "test_txt.txt"

loaded_model = tf.keras.models.load_model(model_path)

with open(txt_path, "r") as f:
	txt = f.read()
	txt = " ".join(txt.split("\n"))
	
make_pred_and_show(loaded_model, txt)