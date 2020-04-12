import os

import torch
import torch.nn as nn

import config

class CheckPoint(object):
    """
    save model state to file
    check_point_params: model, optimizer, epoch
    """

    def __init__(self, save_path):

        self.save_path = os.path.join(save_path, "check_point")
        self.check_point_params = {'model': None,
                                   'optimizer': None,
                                   'epoch': None}

        # make directory
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def load_state(self, model, state_dict):
        """
        load state_dict to model
        :params model:
        :params state_dict:
        :return: model
        """
        # set the model in evaluation mode, otherwise the accuracy will change
        model.eval()
        model_dict = model.state_dict()

        for key, value in list(state_dict.items()):
            if key in list(model_dict.keys()):
                model_dict[key] = value
            else:
                pass
        model.load_state_dict(model_dict)
        return model

    def load_model(self, model_path):
        """
        load model
        :params model_path: path to the model
        :return: model_state_dict
        """
        if os.path.isfile(model_path):
            print("|===>Load retrain model from:", model_path)
            # model_state_dict = torch.load(model_path, map_location={'cuda:1':'cuda:0'})
            model_state_dict = torch.load(model_path, map_location='cpu')
            return model_state_dict
        else:
            assert False, "file not exits, model path: " + model_path

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            print("|===>Load resume check-point from:", checkpoint_path)
            self.check_point_params = torch.load(checkpoint_path)
            model_state_dict = self.check_point_params['model']
            optimizer_state_dict = self.check_point_params['optimizer']
            epoch = self.check_point_params['epoch']
            return model_state_dict, optimizer_state_dict, epoch
        else:
            assert False, "file not exits" + checkpoint_path

    def save_checkpoint(self, model, optimizer, epoch, index=0):
        """
        :params model: model
        :params optimizer: optimizer
        :params epoch: training epoch
        :params index: index of saved file, default: 0
        Note: if we add hook to the grad by using register_hook(hook), then the hook function
        can not be saved so we need to save state_dict() only. Although save state dictionary
        is recommended, some times we still need to save the whole model as it can save all
        the information of the trained model, and we do not need to create a new network in
        next time. However, the GPU information will be saved too, which leads to some issues
        when we use the model on different machine
        """

        # get state_dict from model and optimizer
        model = self.list2sequential(model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.state_dict()
        optimizer = optimizer.state_dict()

        # save information to a dict
        self.check_point_params['model'] = model
        self.check_point_params['optimizer'] = optimizer
        self.check_point_params['epoch'] = epoch

        # save to file
        torch.save(self.check_point_params, os.path.join(
            self.save_path, "checkpoint_%03d.pth" % index))
    
    def list2sequential(self, model):
        if isinstance(model, list):
            model = nn.Sequential(*model)
        return model

    def save_model(self, model, best_flag=False, index=0, tag=""):
        """
        :params model: model to save
        :params best_flag: if True, the saved model is the one that gets best performance
        """
        # get state dict
        model = self.list2sequential(model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.state_dict()
        if best_flag:
            if tag != "":
                torch.save(model, os.path.join(self.save_path, "%s_best_model.pth"%tag))
            else:
                torch.save(model, os.path.join(self.save_path, "best_model.pth"))
        else:
            if tag != "":
                torch.save(model, os.path.join(self.save_path, "%s_model_%03d.pth" % (tag, index)))
            else:
                torch.save(model, os.path.join(self.save_path, "model_%03d.pth" % index))
