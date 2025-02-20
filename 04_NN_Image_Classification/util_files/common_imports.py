# Standard Library Imports
import os

# Scientific and Numeric Libraries
import numpy as np
import pandas as pd
import random

# Image Processing and Computer Vision
import cv2
from PIL import Image
import skimage 
from skimage import morphology, exposure, measure, img_as_ubyte
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, canny
from skimage.color import rgb2gray 
from skimage.filters import meijering, sato, frangi, hessian, sobel, scharr, difference_of_gaussians, window, threshold_otsu
from scipy.fft import fftn, fftshift

# Deep Learning and TensorFlow
import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Sequential, Model 
from keras.layers import ReLU, Conv2D, MaxPooling2D, Conv2DTranspose, Input, Dense
from keras.layers import Reshape, BatchNormalization, Flatten, Dropout, Concatenate, concatenate

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Machine Learning and Dimensionality Reduction
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import umap.umap_ as umap