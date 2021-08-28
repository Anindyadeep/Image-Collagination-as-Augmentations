import os
import torch
from PIL import Image
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = {
    0: 'Damaged_Infrastructure', 
    1: 'Fire_Disaster', 
    2: 'Human_Damage', 
    3: 'Land_Disaster', 
    4: 'Non_Damage', 
    5: 'Water_Disaster'
    }

def test_eval(model, test_path, test_class, transform):
  ans = []
  files = test_class.keys()
  labels = test_class.values()
  num_correct = 0
  for file , idx in zip(files, labels):
      path = os.path.join(test_path, file)
      try:
          img = Image.open(path)
          tensor = transform(img)
          temp_new_path = file.split('.')[0] + '.png'
          new_path = os.path.join(test_path, temp_new_path)
          img.save(new_path)
          img_file = Image.open(new_path).convert('RGB')
          tensor_img = transform(img_file)
          tensor = tensor.unsqueeze(0).to(device)
          output = model(tensor)
          confidence, prediction = torch.max(output, 1)
          ans.append(prediction.tolist()[0])
          if prediction == idx:
            num_correct += 1
      except RuntimeError as e:
          print('could not infer dtype of JPEG')
  return ans, (num_correct/len(test_class))

def _get_test_reults(model, transform, test_path, save_as):
    preds = []
    preds_class = []

    files = os.listdir(test_path)
    for file in files:
        jpeg_path = os.path.join(test_path, file)
        path = os.path.join(test_path, file)
        try:
            img = Image.open(path)
            #temp_new_path = file.split('.')[0] + '.png'
            #new_path = os.path.join(test_path, temp_new_path)
            new_path = path
            img.save(new_path)
            img_file = Image.open(new_path).convert('RGB')
            tensor_image = transform(img_file)
            tensor = tensor_image.unsqueeze(0).to(device)
            output = model(tensor)
            _, prediction = torch.max(output, 1)
            preds.append(prediction.tolist()[0])
            preds_class.append(classes[prediction.tolist()[0]])
        except RuntimeError as e:
            print('could not infer dtype of JPEG, skipping ...')
    
    if save_as:
        save_dict = {
            'image_name: ': files,
            'predicted_class_idx': preds,
            'predicted_class_label': preds_class}
        df = pd.DataFrame(save_dict)
        df.to_csv(save_as)
    return preds