# Author: FLDCLA001, GTTDAN002, NDHCAR002

## Set Up information:
Setup within a virtual environment on windows is as follows.
Assuming that python and python-venv installed.

1) To build the virtual environment and install necessary packages using a python script:

```
> python MakeWindows.py
```

2) To activate the virtual environment:

```
> .\venv\Scripts\activate
```

3) To install requirements

```
pip install -r requirements.txt
```

4) To run the skeleton code:

- To run the BASELINE classifier
```
> python Classifier_1.py
```
- To run the IMPROVED classifier
```
> python Classifier_2.py
```
- To train the model
```
Select "Y" when prompted
Any other key will directly download the model.pth file from the directory
```
- To download the datasets from Kaggle 
***Note*** Either option will take a minute or two, it is a large amount of data. Thankfully, you only have to select "Y" once.
When the shark data is loaded into the directory, it can be used numerous times by either Classifier.
```
https://www.kaggle.com/datasets/clairefielden/sharks
Ensure each class (eg. basking_resized) is in its designated folder, stored in ./sharks/sharks
```
- Alternatively, one can select "Y" when prompted. This will download the datset directly into the python program. 
```
Ensure the "kaggle.json" file in the directory is installed in the following folder on your computer: C:\Users\User\.kaggle
```
- Making a prediction

```
When the model has been trained (after 10 epochs), an image of a shark will come up. (This is a lemon shark).
Exit the image and the system will output the classification of the shark.
```
- Accessing only the models

```
In order to view only the topology of Classifier_1, please see model1.pth
The same applies for model2.pth
```


