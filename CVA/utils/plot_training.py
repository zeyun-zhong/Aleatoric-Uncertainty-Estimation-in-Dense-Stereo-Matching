from matplotlib import pyplot as plt
import numpy as np
import math
import os
import sys
sys.path.insert(1, '../CVA-Net')


def smooth_loss(d_diff):
    s = np.arange(-1, 1, 0.1)
    r = np.sqrt(3) * np.exp(s)
    tmp = np.abs(d_diff - r)
    loss = np.where(tmp < 1, 0.5 * np.square(d_diff - r), np.abs(d_diff - r) - 0.5)
    return s, loss


def visualize_loss():
    s, loss = smooth_loss(2)
    plt.plot(s, loss)
    plt.title('Visualization of loss w.r.t. uncertainty')
    plt.ylabel('Loss')
    plt.xlabel('Uncertainty')
    plt.savefig('visu.svg')
    plt.show()


def plot_loss(train, val, models, save_name, savefig):
    small_size = 10
    medium_size = 12
    big_size = 14

    font = {
        'family': 'serif',
        'weight': 'normal',
        'size': big_size,
    }

    plt.figure(figsize=(8, 6))
    for i, (train_loss, val_loss) in enumerate(zip(train, val)):
        epoch = range(1, len(train_loss) + 1)
        model_name = models[i].replace('CVA-Net_', '')
        if '+' in model_name: model_name = model_name.split('+')[0]
        plt.plot(epoch, train_loss, label='train '+model_name)
        plt.plot(epoch, val_loss, label='val '+model_name)
    plt.title('Training', fontdict=font)
    font['size'] = medium_size
    plt.xlabel('epoch', fontdict=font)
    plt.ylabel('loss', fontdict=font)
    font['size'] = small_size
    plt.legend(prop=font)
    # plt.xlim([0, len(train_loss)+1])
    # plt.ylim([0, math.ceil(max(train_loss))])
    plt.xlim([0, 40])
    plt.ylim([4, 12])
    plt.xticks(fontsize=small_size)
    plt.yticks(fontsize=small_size)

    if savefig: plt.savefig(save_name + '.svg', bbox_inches='tight')
    plt.show()


def read_txt(path, folders):
    train, val = [], []
    for folder in folders:
        folder = folder.split('+') if '+' in folder else [folder]
        train_loss, val_loss = [], []
        for fold in folder:
            txt_path = os.path.join(path, fold, 'loss.txt')
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.split(', ')
                    if 'train' in line: continue
                    train_loss += float(line[0]),
                    val_loss += float(line[1]),
        train += train_loss,
        val += val_loss,
    return train, val


if __name__ == '__main__':
    path = '/home/zeyun/Projects/CVA/experiments/dynamic-depth/'
    models = ['CVA-Net_Uniform_FC_16', 'CVA-Net_Uniform_FC_32+CVA-Net_Uniform_FC_32_2',
              'CVA-Net_Uniform_l1_FC_16+CVA-Net_Uniform_l1_FC_16_2',
              'CVA-Net_Uniform_l1_FC_32+CVA-Net_Uniform_l1_FC_32_2']
    train, val = read_txt(path, models)
    plot_loss(train, val, models, save_name='Training', savefig=True)
