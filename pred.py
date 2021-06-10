import numpy as np
import torch
from torch.autograd import Variable
from data import save_nii


def prediction(dataloader, model, device, save_pred_path=False):
    patient_ids = []
    real_y_list = []
    fake_y_list = []

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            real_x = Variable(data['x']).to(device)
            real_y = Variable(data['y']).to(device)
            # if cuda:
            #     real_x = real_x.to(device)
            #     real_y = real_y.to(device)

            fake_y = model(real_x)
            if len(fake_y) != 1:  # model output: enc_list, out
                fake_y = fake_y[-1]

            real_y = real_y.cpu().detach().numpy()
            fake_y = fake_y.cpu().detach().numpy()

            real_y = np.squeeze(real_y)
            fake_y = np.squeeze(fake_y)

            real_y_list.append(real_y)
            fake_y_list.append(fake_y)

            patient_id = str(data['patient_id'][0])
            patient_ids.append(patient_id)

            if save_pred_path:
                # save_nii(real_y, f'{save_pred_path}{patient_id}_real_y')
                save_nii(fake_y, f'{save_pred_path}{patient_id}_fake_y')

    return real_y_list, fake_y_list, patient_ids


def prediction_self(dataloader, model, device, save_pred_path=False):
    patient_ids = []
    real_y_list = []
    self_y_list = []

    for idx, data in enumerate(dataloader):
        real_y = Variable(data['y']).to(device)

        self_y = model(real_y)

        if len(self_y) != 1: # model output: enc_list, out
            self_y = self_y[-1]

        real_y = real_y.cpu().detach().numpy()
        self_y = self_y.cpu().detach().numpy()

        real_y = np.squeeze(real_y)
        self_y = np.squeeze(self_y)

        real_y_list.append(real_y)
        self_y_list.append(self_y)

        patient_id = str(data['patient_id'][0])
        patient_ids.append(patient_id)

        if save_pred_path:
            # save_nii(real_y, f'{save_pred_path}{patient_id}_real_y')
            save_nii(self_y, f'{save_pred_path}{patient_id}_self_y')

    return real_y_list, self_y_list, patient_ids