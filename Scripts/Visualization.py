import h5py
import pickle
import numpy as np
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

def train_test_plot_CO2(X_train, y_train, X_test, y_mc, y_mc_std):
    plt.plot(X_train.squeeze(), y_train.squeeze(), 'g', alpha=0.8);
    plt.plot(X_test.squeeze(), y_mc, alpha=0.8);
    plt.axvline(X_train.squeeze()[-1], color='g',linestyle='--');
    plt.fill_between(X_test.squeeze(), y_mc-2*y_mc_std, y_mc+2*y_mc_std, alpha=0.3);
    plt.title('MC dropout with ReLU non-linearities');
    plt.ylim([-20, 20]);
    
def rotate_image_one(X1):
    plt.figure(figsize=(8, 1))

    gs = gridspec.GridSpec(1, 12)
    gs.update(wspace=0, hspace=0)

    for i in range(len(X1)):
        plt.subplot(gs[i])
        plt.imshow(X1[i].squeeze(), cmap='gray');
        plt.axis('off');
        
def softmax_scatter(y1_so,y1_si):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(np.tile(np.arange(1, 13), y1_si.shape[0]), y1_si[:, :, 1].flatten(), \
                color='g', marker='_', linewidth=None, alpha=0.5, label='1');
    plt.scatter(np.tile(np.arange(1, 13), y1_si.shape[0]), y1_si[:, :, 7].flatten(), \
                color='r', marker='_', linewidth=None, alpha=0.5, label='7');
    plt.scatter(np.tile(np.arange(1, 13), y1_si.shape[0]), y1_si[:, :, 5].flatten(), \
                color='b', marker='_', linewidth=None, alpha=0.5, label='5');
    plt.title('Softmax input scatter');
    plt.legend(framealpha=0.7);
    
    plt.subplot(1, 2, 2)
    plt.scatter(np.tile(np.arange(1, 13), y1_so.shape[0]), y1_so[:, :, 1].flatten(), \
                color='g', marker='_', linewidth=None, alpha=0.5, label='1');
    plt.scatter(np.tile(np.arange(1, 13), y1_so.shape[0]), y1_so[:, :, 7].flatten(), \
                color='r', marker='_', linewidth=None, alpha=0.5, label='7');
    plt.scatter(np.tile(np.arange(1, 13), y1_so.shape[0]), y1_so[:, :, 5].flatten(), \
                color='b', marker='_', linewidth=None, alpha=0.5, label='5');
    plt.title('Softmax output scatter');
    plt.legend(framealpha=0.7);
    
    plt.tight_layout();

def plot_grid(rows, cols, figsize, image_root_path, labels, data_shape):
    f, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=figsize)

    for ax, label, name in zip(axes.ravel(), labels['Label'], labels['Common Name']):
        img = np.random.choice(os.listdir(os.path.join(image_root_path, label)))
        img = Image.open(os.path.join(image_root_path, label, img))
        img = img.resize(data_shape)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("{}: {}".format(label, name))

def visualize_probdist(pred_no, classes, pred_bayes_dist, images, labels, label_mapping, ood_mapping=None):
    test_id = np.random.randint(0, high=len(labels), size=(pred_no,))
    pred_bayes_dist=np.transpose(np.array(pred_bayes_dist),(1,0,2))
    f, axes = plt.subplots(len(test_id), 2, figsize=(10, 4*pred_no))
    f.tight_layout(h_pad=5, w_pad=0)
    axs = axes.ravel()

    ax_idx = 0
    for tid in test_id:
        current_ax = axs[ax_idx]
        for i in range(classes):
            current_ax.hist(pred_bayes_dist[tid][:, i], alpha=0.3, label=label_mapping[i])
            current_ax.axvline(np.quantile(pred_bayes_dist[tid][:, i], 0.5), color='red', linestyle=':', alpha=0.4)
            current_ax.axvline(0.5, color='green', linestyle='--')
            current_ax.legend()
            current_ax.set_xlabel('Probability')
            current_ax.set_ylabel('Count')
            if ood_mapping is not None:
                current_ax.title.set_text("Correct Label: {}".format(ood_mapping[labels[tid]]))
            else:
                current_ax.title.set_text("Correct Label: {}".format(label_mapping[labels[tid]]))
        np.set_printoptions(False)
        ax_idx += 1
        current_ax = axs[ax_idx]
        current_ax.axis('off')
        current_ax.title.set_text("For Test Image Index: {}".format(tid))
        current_ax.imshow(images[tid])
        ax_idx += 1