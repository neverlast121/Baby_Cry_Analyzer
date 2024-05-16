import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
from models import *
from pytorch_utils import move_data_to_device
import config


def audio_prediction(audio_path):
    """Inference audio prediction result of an audio clip.
    """
    # parameters
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    model_type = config.model_type

    checkpoint_path1 = config.checkpoint_path1
    checkpoint_path2 = config.checkpoint_path2
    
    classes_num = config.classes_num
    labels = config.labels

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Model 1
    Model1 = eval(model_type)
    model1 = Model1(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=527)
    
    checkpoint1 = torch.load(checkpoint_path1, map_location=device)
    model1.load_state_dict(checkpoint1['model'])

    # Model 2
    Model2 = eval(model_type)
    model2 = Model2(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint2 = torch.load(checkpoint_path2, map_location=device)
    model2.load_state_dict(checkpoint2['model'])

    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model1.eval()
        batch_output_dict1 = model1(waveform, None)

    clipwise_output1 = batch_output_dict1['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output1)[::-1]
    top_five = sorted_indexes[:5]

    if 17 in top_five:
        list_d = [0,0,0,0,1,0,0,0,0]
        predictions = ['{}: {:.3f}'.format(np.array(labels)[k], list_d[k]) for k in range(9)] 
        response = {'mode': "Don't know", 'result' : [0,0,0,0,1,0,0,0,0], 'predictions': predictions}
        return responce
    
    elif 23 in top_five:
        with torch.no_grad():
            model2.eval()
            batch_output_dict2 = model2(waveform, None)
        clipwise_output2 = batch_output_dict2['clipwise_output'].data.cpu().numpy()[0]
        sorted_indexes = np.argsort(clipwise_output2)[::-1]
        predictions = ['{}: {:.3f}'.format(np.array(labels)[k], clipwise_output2[k]) for k in range(9)]
        response = {'mode': 'Baby is Crying', 'result': clipwise_output2.tolist(), 'predictions': predictions}
        return response
    
    else:
        list_d = [0,0,0,0,1,0,0,0,0]
        predictions = ['{}: {:.3f}'.format(np.array(labels)[k], list_d[k]) for k in range(9)] 
        response = {'mode': "Don't know", 'result' : [0,0,0,0,1,0,0,0,0], 'predictions': predictions}
        return response


if __name__ == '__main__':
    audio_path = '5afc6a14-a9d8-45f8-b31d-c79dd87cc8c6-1430757039803-1.7-m-48-bu.wav'
    responce = audio_prediction(audio_path)
    print(responce)