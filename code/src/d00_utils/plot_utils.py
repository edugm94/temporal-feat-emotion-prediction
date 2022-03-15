# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.12.14
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden

import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, cms,  classes, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
        cm: confusion matrix in perecentage
        cms: standard deviation error from cm
        classes: Name of each label
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # In case standar error wants to be displayed
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(cms[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center", fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")

            # In case only percentage is displayed
            #plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '%',
            #         horizontalalignment="center",
            #         verticalalignment="center", fontsize=15,
            #         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')