# Machine Learning Portfolio

**Author: Levi Guerra** ([about me](#About-me))

Compilation of my **public Machine Learning projects**. They come in the shape of *Jupyter Notebooks* (Python code with text and images). This portfolio contains projects for different types of ML-algorithms (Supervised, Unsupervised and Deep Learning) and a final section with larger and more ambitious personal projects. 

Most of the projects include a thorough **Exploratory Data Analysis** (libraries for visualization: *matplotlib*, *mpl_toolkits*, *seaborn*). Although this repository does not have educational purposes, due to its organization and how clear the code and the explanations are, it could have didactical value. 

*Libraries: NumPy, SciPy, Pandas, Random, Collections, Scikit-Learn, ImbLearn, Tensorflow, Keras*.

### Contents

- [Supervised & Unsupervised Learning](#Supervised-and-Unsupervised-Learning)
- [Deep Learning](#Deep-Learning)
- [Others](#Others)
- [Personal Projects](#Personal-Projects)

------

## Supervised and Unsupervised Learning
  - ### Regression Models
    - [Linear Regression - Boston Housing Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/01_Linear-Regression_Boston-Housing-Dataset.ipynb): Prediction of the median value of a house in Boston using a dataset with information about the housing in the city. Simple and multiple linear regressions will be presented and compared to scikit-learn regressions.
    - [Polynomial Regression - Insurance Claims Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/02_Polynomial-Regression_Insurance-Dataset.ipynb): Predicting the number of insurance claims in a company by means of the information collected from previous years. The method presented will be compared to the *Scikit-Learn* implementation.
    - [Logistic Regression - Breast Cancer Wisconsin Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/03_Logistic-Classification_Breast-Cancer-Wisconsin-Dataset.ipynb): The Breast Cancer Wisconsin dataset contains biophyisical information about different samples of benign and malignant cells. A logistic regression is used to predict whether a cell is of one type or another, based on its characteristic.
    - [Regression Models - MNIST Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/07_Regression-Models_MNIST.ipynb): Classifying handwritten numbers (MNIST) by means of different types of regression. The accuracy between methods can be compared.
  - ### K-Nearest Neighbors
    - [KNN Classification - Wine Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/04_KNN-Clasification_Wine-Dataset.ipynb): Classifying wines of unknown type based on the similarity (*nearest neighbor*) with other labelled wines.
  - ### K-Means Clustering
    - [K-Means Clustering - Iris Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/05_K-Means-Clustering_Iris-Dataset.ipynb): Clustering unlabelled flowers of *k* different types. The optimal number of clusters k is also found.
    - [An Unsupervised Approach to MNIST Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/06_An-Unsupervised-Approach-to-MNIST.ipynb): Clustering of handwritten digits (MNIST) is performed assuming we don't know how many different types of digits there are.

## Deep Learning
  - ### Multilayer Perceptron (MLP)
    - [Multilayer Perceptron - MNIST Dataset](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/08_MLP_MNIST.ipynb): Classifying handwritten digits (MNIST) by means of neural networks. Different multilayer perceptron architechtures and optimizers are used and compared. *Keras* library used.
  - ### Convolutional Neural Networks (CNN)
    - [Convolutional Neural Networks - MNIST dataset](): Being updated!
    - [Convolutional Neural Networks - CIFAR-10 dataset](): Being updated!
  - ### Autoencoders
    - [Convolutional Autoencoders - MNIST dataset](): Being updated!
    - [Convolutional Autoencoders - Tiny ImageNet dataset](): Being updated!
    
## Others
 - ### Data Augmentation
    - [Data Augmentation on Images](https://github.com/LeviGuerra/Machine-Learning-Portfolio/blob/master/Codes_and_Datasets/09_Data-Augmentation_MNIST-Dataset.ipynb): The amount of data to train a model might not be enough to properly generalize the features of the classes to learn. Here we show how data augmentation (rotations, translations and noise) performed over the MNIST Dataset is necessary if we want a more general, robust model.

## Personal Projects
This section is a connection to other larger personal projects. Due to their more ambitious nature they are presented in individual repositories.

- **[Customer Clustering and Churn Prediction in a Bank](https://github.com/LeviGuerra/Bank-Churn-Prediction/blob/master/Code_and_Dataset/Customer_Clustering_and_Churn_Prediction.ipynb)**: In this project a solution for the customer churn in a bank is presented. The model can be divided into two goals: split customers into groups of highly similar members (clustering) and predict by means of deep learning the churn risk of individual customers. The application of the model here presented would translate in a positive economical impact for the institution. Furthermore these results are not exclusive for banks: any type of institution/company with a large number of customers (or employees), where loyalty represents a valuable feature, could benefit from it.

------

## About me:

Levi Guerra García (26 y/o). Physicist (M.Sc.) passionate about Machine Learning, Data Analysis and Neuroscience. Further information about myself [here](https://www.linkedin.com/in/leviguerra/).

For any suggestion regarding the content of the portfolio, or for a collaboration proposal, please do not hesistate to send me an email: leviguerra.g@gmail.com
