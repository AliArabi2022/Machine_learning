ResNet Transfer Learning Tutorial on CIFAR-10
This repository contains a Jupyter Notebook that provides a step-by-step tutorial on Transfer Learning using ResNet architectures in PyTorch. The tutorial is designed for beginners to intermediate learners, assuming minimal prior knowledge, and focuses on explaining concepts simply while reproducing key experiments from the original ResNet paper.
Project Overview
The notebook explores the history of CNNs, the motivation behind ResNet (residual learning to solve the degradation problem), and practical implementation of Transfer Learning. It uses the CIFAR-10 dataset to demonstrate:

Training plain (non-residual) networks vs. residual networks from scratch at different depths (e.g., ResNet-20, ResNet-32).
Comparing performance curves to show how residuals enable deeper networks without accuracy degradation.
Applying Transfer Learning: Fine-tuning a pre-trained ResNet-18 (on ImageNet) vs. training from scratch.

Key features:

Historical background on CNN evolution (Perceptron, LeNet, AlexNet, VGG, GoogLeNet).
Implementation of BasicBlock (plain), ResidualBasicBlock, and Bottleneck blocks.
Factory functions for easy creation of ResNet variants (e.g., ResNet-20, ResNet-50 for CIFAR).
Training loops, visualization of curves, and comparisons using matplotlib.
Transfer Learning with torchvision's pre-trained models.

The goal is to help users understand why ResNet revolutionized deep learning and how to apply it to new tasks with limited data.