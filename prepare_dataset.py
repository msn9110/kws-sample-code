import os
import glob
import random
from tqdm import tqdm
import hashlib

from phoneme import zelements
import my_signal

# random.seed(119)
with open('select.txt') as f:
    selected_labels = set([lbl for lbl in f.read().split('\n')
                           if lbl])


def get_files(root):
    u_files = []
    v_files = []
    for p in glob.glob(root + '/*/*.wav'):
        if p.split('/')[-2] in selected_labels:
            v_files.append(p)
        else:
            u_files.append(p)
    return v_files, u_files


class DataProcessor:

    def __init__(self, root, testing_root, model_setting, val_percentage=10,
                 salt=0, oversampling=False):
        mode = 0  # indicates a pronunciation of zhuyin

        from phoneme import optionals as aux_labels
        self.model_settings = model_setting
        files, u_files = get_files(root)
        t_data = {}
        v_data = {}
        m_data = [t_data, v_data]
        base = int(100 - val_percentage)
        labels = set()
        for p in tqdm(files):
            path = os.path.abspath(p)
            label, name = p.split('/')[-2:]
            sha1 = hashlib.sha1()
            sha1.update(name.encode('utf-8'))
            which = ((int(sha1.hexdigest(), 16) + salt) % 100 + 1) // base
            elements = zelements[label]
            my_label = ([label] + elements)[mode]
            dataset = m_data[which]
            if my_label not in dataset:
                dataset[my_label] = []
            dataset[my_label].append(path)
            labels.add(my_label)

        if oversampling:
            for dataset in m_data[:1]:
                max_amount = max(map(len, dataset.values()))
                for l, fs in dataset.items():
                    offset = max_amount - len(fs)
                    random.shuffle(fs)
                    dataset[l] += fs[:offset]

        labels = aux_labels[mode] + list(sorted(labels))
        label_index = {l: i for i, l in enumerate(labels)}
        training_set = [(l, p) for l, fs in m_data[0].items()
                        for p in fs]
        background_noise = []
        for p in u_files:
            p = os.path.abspath(p)

            if os.path.split(os.path.dirname(p))[1] == '_background_noise_':
                audio = my_signal.decode_wav(p).audio
                background_noise.append(audio.numpy().flatten())
            else:
                training_set.append((aux_labels[mode][0], p))
        if not background_noise:
            raise FileNotFoundError('Not found any noise')
        self.background_data = background_noise

        # silence file, it will be multiply with 0 during processing
        training_set.extend([(aux_labels[mode][1], training_set[0][1])] * int(len(files) / 20))
        validation_set = [(l, p) for l, fs in m_data[1].items()
                          for p in fs]
        testing_set = []
        for i, fs in enumerate(get_files(testing_root)):
            for p in fs:
                p = os.path.abspath(p)
                label = aux_labels[mode][0]
                if not i:
                    l = p.split('/')[-2]
                    label = ([l] + zelements[l])[mode]
                testing_set.append((label, p))

        self.label_index = label_index
        self.labels = labels
        self.set_index = {'training': training_set, 'validation': validation_set, 'testing': testing_set}

    def get_data(self, mode, batch_size):
        dataset = self.set_index[mode]
        print(mode, len(dataset))
        return dataset, batch_size, self.background_data, mode, self.model_settings, self.label_index

    def set_size(self, which):
        return len(self.set_index[which])
