import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

class Grad_tb_callback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, histogram_freq, val_dataset):
        super(Grad_tb_callback, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)

    def _log_gradients(self, epoch):
        writer_weight = tf.summary.create_file_writer(
            os.path.join(self.log_dir, 'weights'))  # self._get_writer(self._train_run_name)
        writer_grad = tf.summary.create_file_writer(os.path.join(self.log_dir, 'grads'))

        with writer_grad.as_default(), tf.GradientTape() as g:
            # take a batch from the validation dataset
            for inputs, labels in self.val_dataset.take(1):  # only take first element of dataset
                inputs = inputs
                labels = labels
                labels = tf.cast(labels, dtype=tf.float32)

            # forward pass through the model
            y_pred = self.model(inputs)

            # calculate the loss
            loss_value = self.model.compiled_loss(labels, y_pred)

            # calculate the gradients
            gradients = g.gradient(loss_value, self.model.trainable_weights)

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)
                tf.summary.scalar(
                    weights.name.replace(':', '_') + '_grad_norm', data=tf.norm(grads), step=epoch)
                print(f"{weights.name.replace(':', '_')}_grad_norm: {tf.norm(grads)}")
        writer_grad.flush()

        with writer_weight.as_default():
            # record the weights
            for weights in self.model.trainable_weights:
                tf.summary.histogram(
                    weights.name.replace(':', '_'), data=weights, step=epoch)
        writer_weight.flush()


def monitor_weights_and_grads(model, val_dataset, epoch, logdir):
    # Use GradientTape to record gradients
    with tf.GradientTape() as tape:
        # #get the validation data todo only take 1 batch
        for inputs, labels in val_dataset.take(1):  # only take first element of dataset
            inputs = inputs
            labels = labels
            labels = tf.cast(labels, dtype=tf.float32)

        # forward pass through the model
        y_pred = model(inputs)

        # calculate the loss
        loss_value = model.loss(labels, y_pred)

    # Get the gradients of the loss with respect to the variables (same format as model.trainable_weights
    model_weights = model.trainable_weights
    gradients = tape.gradient(loss_value, model_weights)

    # compute some usefull numbers
    nr_weights = len(model_weights)
    nr_layers = len(model.layers)
    weights_per_gnn_layer = int(nr_weights / (nr_layers - 1))
    print(f'nr layers: {nr_layers} - weight matrices per gnn layer: {weights_per_gnn_layer}')

    # todo doesnt work for skip layers
    # loop over model weights
    for i, (weight, grad) in enumerate(zip(model_weights, gradients)):
        # print(f'weight dir: {dir(weight)}')
        plt.subplot(nr_layers - 1, weights_per_gnn_layer, i + 1)
        plt.hist(weight.numpy().flatten())
        plt.gca().set_title(weight.name)
    plt.suptitle(f'WEIGHTS')
    fig = plt.gcf()
    fig.set_size_inches(weights_per_gnn_layer * 4, (nr_layers - 1) * 4)
    fig.savefig(os.path.join(logdir, f'weights_epoch_{epoch}.pdf'))
    # fig.savefig(os.path.join(logdir, f'weights_epoch_{epoch}.png'))
    plt.show()

    # loop over model weights and gradients
    weight_names = []
    for i, (weight, grad) in enumerate(zip(model_weights, gradients)):
        # print(f'weight dir: {dir(weight)}')
        plt.subplot(nr_layers - 1, weights_per_gnn_layer, i + 1)
        plt.hist(grad.numpy().flatten(), color='red', ec='red')
        plt.gca().set_title(f'grad_{weight.name}')
    plt.suptitle(f'GRADIENTS')
    fig = plt.gcf()
    fig.set_size_inches(weights_per_gnn_layer * 4, (nr_layers - 1) * 4)
    fig.savefig(os.path.join(logdir, f'grads_epoch_{epoch}.pdf'))
    # fig.savefig(os.path.join(logdir, f'weights_epoch_{epoch}.png'))
    plt.show()

    # get weights by name:
    # for i, layer in enumerate(model.layers):
    #     print(f'layer: {i}')
    #     print(f'layer:{layer._name}')
    #     #print(f'attributes of layer: {dir(layer)}')
    #     #print(f'weights: {layer.get_weights()}')
    #     if 'gnn_layer' in layer._name:
    #         # print weights
    #         print(f'edge weights: {layer.edge_weights}')
    #         print(f'm weights: {layer.m_weights}')
    #         print(f'k weights: {layer.k_weights}')
    #         for j in range(3):
    #             print(f'layer {i}) - j: {j}) - 3i + j: {3*i + j})')
    #             print(model_weights[3*i + j])
