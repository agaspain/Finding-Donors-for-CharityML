# Finding Donors for Charity using Supervised Learning

This project was completed as a part of the Data Science Nanodegree from Udacity.

## Problem statement

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity.

Our goal in this project is to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent. Thus, during this project we’ll se how we can accurately model individuals' income using data collected from the 1994 U.S. Census, in order to predict whether an individual makes more than $50,000.

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the 'fnlwgt' feature and records with missing or ill-formatted entries.

## Summary of the proposed approach

During the data preparation and transformation of this project, I describe how we can deal with categorical and numerical features. In addition, since this project is characterised by being imbalance, I show how we can under-sample our dataset with the **Tomek link method**. Then, during the model exploration, I explain how we can set our baseline method and how we can choose the best **scoring metrics** based on the characteristics of our data. In this phase of the project, I train and evaluate some **supervised machine learning algorithms** (such as GaussianNB, SVC, Adaboost) comparing their accuracy and speed. I also explore the impact of reducing number of features in data. Based on the results of the analysis, I choose the 3 best systems that provides the highest donation yield and I proceed to fine tune their parameters. Finally, I describe how we can build new model by combining the prediction from the 3 best predictive models using the **stacking method**.

## Software and Libraries

This project uses the following software and Python libraries:

- [Python](https://www.python.org/downloads/release/python-364/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [Matplotlib](https://matplotlib.org/)
- [imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).


