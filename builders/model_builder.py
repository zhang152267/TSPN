from model.TSPN import TSPN

#from model.sanet_S import DualResNet
def build_model(model_name, num_classes):
    if model_name == 'TSPN':
        return TSPN(classes=num_classes)