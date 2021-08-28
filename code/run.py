import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from process_test_file import test_eval, _get_test_reults
import warnings
warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_path = str(os.getcwd())[:-4] + 'test_folder'
state_dict_path =  os.path.join(os.getcwd()[:-5], 'models/VGG19.pth')


test_class = {'in1.jpg'    : 0,
              'land3.jpg'  : 3,
              'in3.jpeg'   : 0,
              'earth1.jpeg': 0,
              'd1.jpeg'    : 3,
              'urb3.jpeg'  : 1,
              'd2.jpeg'    : 3,
              'urb2.jpeg'  : 1,
              'earth3.jpeg': 0,
              'land2.jpeg' : 3,
              'sea1.jpg'   : 4,
              'wild2.jpeg' : 1,
              'sea3.jpeg'  : 4,
              'earth4.jpeg': 0,
              'in2.jpeg'   : 0,
              'd4.jpg'     : 3,
              'earth5.jpeg': 0,
              'earth2.jpg' : 0,
              'd3.jpg'     : 3,
              'urb1.jpeg'  : 1,
              'sea2.jpeg'  : 5,
              'urb4.jpeg'  : 1,
              'wild3.jpeg' : 1,
              'wild1.jpg'  : 1,
              'land1.jpg'  : 3}


transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((224,224)),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                ])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 6
model = models.vgg19(pretrained=False)
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))


def get_accuracy_test_results():
    print('USING BEST MODEL: VGG19 ...')
    ans, acc = test_eval(model, test_path, test_class, transform)
    print("Accuracy of the test path: ", acc*100, "%")
    print("Predicted values: ", ans)



def get_test_results(test_path, save_as='results.csv'):
    _get_test_reults(model, transform, test_path, save_as=save_as)
    print('process finished ...')

