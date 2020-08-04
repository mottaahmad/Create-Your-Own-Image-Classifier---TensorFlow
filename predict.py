import tensorflow_hub as hub
import tensorflow as tf

import numpy as np


from PIL import Image
import argparse
import json


# initialize the parser
parser = argparse.ArgumentParser (description = 'Parser of predict.py script')
# define Mandatory and Optional Arguments for the script
parser.add_argument ('image_path', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_model', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int, required = False)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str, default = 'label_map.json', required = False)
# parse the arguements
args = parser.parse_args()
#print predict(args.image_path,args.load_model,args.top_k,args.category_name)

#loading JSON file

with open(args.category_name, 'r') as f:
    class_names = json.load(f)
    class_names_dict = dict()
    for i in class_names:
        class_names_dict[str(int(i)-1)] = class_names[i]
        
        
#loading model from checkpoint provided
model = tf.keras.models.load_model(args.load_model, custom_objects={'KerasLayer':hub.KerasLayer})
print(model.summary())
    
#to do: use the argument names in the functions

# function to process a PIL image for use in a Tensorflow model
image_size = 224
def process_image(image):
    
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size,image_size))

    image /= 255
    return image


# TODO: Create the predict function

def predict(image_path, model, top_k):
    
    image = Image.open(image_path)
    test_image = np.asarray(image)

    test_image = process_image(test_image)

    print(test_image.shape, np.expand_dims(test_image,axis=0).shape)
    prob_pred = model.predict(np.expand_dims(test_image,axis=0))
    prob_pred = prob_pred[0].tolist()

    values, indices= tf.math.top_k(prob_pred, k=top_k)
    probs=values.numpy().tolist()
    classes=indices.numpy().tolist()

    calsses = args.category_name
    print(probs, class_names)





        
        
if __name__=='__main__':
    predict(args.image_path, args.load_model, args.top_k)