# Required by default for compatibility

sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
model_type = 'Cnn14_DecisionLevelAtt'

checkpoint_path1 = 'checkpoints/Classify_audio.pth'
checkpoint_path2 = 'checkpoints/Baby_cry_cls.pth'

clip_samples = sample_rate * 6    # Audio clips are 6-second

labels = ['bp', 'bu','ch','dc','dk','hu','lo','sc','ti']

classes_num = len(labels)