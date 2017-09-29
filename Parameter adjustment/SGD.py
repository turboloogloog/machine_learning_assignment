import numpy as np
import pandas as pd
import collections
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from math import ceil
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import itertools
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# To calculate relevance betweeb each category in a 'categorical column' and 'income' column
def relevance(label, condition): 
	label = list(label)
	condition = list(condition)
	# Calculate frequency for each category
	word_count = collections.defaultdict(int)
	for word in label:
		word_count[word] += 1

	perc_count = collections.defaultdict(int)
	for index, word in enumerate(label):
		if condition[index] == 1:
			perc_count[word] += 1

	keys = list(word_count.keys())
	contrib = {}
	for i in range(len(keys)):
		contrib[keys[i]] = perc_count[keys[i]] / word_count[keys[i]]

	return contrib
   
# Function to replace categorical column with numerical according to relevance
def CatgReplace(CatgCol, df):
	for col in CatgCol:   
	    col_array = np.array(df[col])
	    col_relv = relevance(col_array, df['income'])
	    col_replc = [col_relv[col_array[x]] for x in range(len(col_array))]
	    df[col] = col_replc
	return df

# Function to replace a numerical column with classified relevance values
def NumReplace(df, name, bins):
	seri_array = np.array(df[name])
	for i, num in enumerate(seri_array):
		label = ceil(num/bins)
		seri_array[i] = label
	seri_relv = relevance(seri_array, df['income'])
	seri_replc = [seri_relv[seri_array[x]] for x in range(len(seri_array))]
	df[name] = seri_replc
	return df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')	
 
#-------------------------------------------------------------------------------------------
# Main Program
#-------------------------------------------------------------------------------------------
# Data preprocessing

filepath = 'adult.data' 
df_header = ['age','workclass','fnlwgt','education','education_num',
                    'marital_status','occupation','relationship','race','sex',
                    'capital_gain','capital_loss','hours_per_week','native_country','income']
# Load original data
adult_df = pd.read_csv(filepath, names = df_header)
# Drop the repeated column
adult_df = adult_df.drop('education_num', axis=1)

# Convert income to 0 and 1 according to ' <=50K' and ' >50K' respectively
adult_df['income'] = adult_df['income'].replace([' <=50K', ' >50K'], [0, 1])

# Buid a traning set
train_df = adult_df[:]

# Extract 'Categorical Columns'
CatgCol = ['workclass','education','marital_status',
           'occupation','relationship','race','sex','native_country'] 
 
# Replace 'Categorical Columns' with numerical values
train_df = CatgReplace(CatgCol, train_df)        

# Extract 'Numerical Columns'
NumCol = ['age','fnlwgt', 'capital_gain','capital_loss','hours_per_week']
# Corresponding bin size
#binSize = [1,1,1,1,1]
binSize = [2, 50, 30, 30, 3]

# Numerical column replacement
for i, names in enumerate(NumCol):
	train_df = NumReplace(train_df, name = names, bins = binSize[i])

train_df = train_df.drop('income', axis = 1)
train_x = train_df.as_matrix()
train_y = adult_df['income'].values

# Load test data
filepath_test = 'adult.test'
test_df = pd.read_csv(filepath_test, names = df_header, skiprows = [0])
test_df['income'] = test_df['income'].replace([' <=50K.', ' >50K.'], [0, 1])
test_df = test_df.drop('education_num', axis=1)
test_df = CatgReplace(CatgCol, test_df)

for i, names in enumerate(NumCol):
	test_df = NumReplace(test_df, name = names, bins = binSize[i])

test_y = test_df['income'].values
test_df = test_df.drop('income', axis = 1)
test_x = test_df.as_matrix()

#'''
#--------------------------------------------------------------------------------------------------
# Classification 
#--------------------------------------------------------------------------------------------------
class_names = ["Greater than 50K", "Less than 50K"]

#---------------------------------------------------------------------------------------------------------------
# Stochastic Gradient Descent
SGD = SGDClassifier(alpha = 0.01, loss="hinge", penalty="elasticnet")#(loss="log", penalty="l1") 0.73 0.97 0.91
                                                         #(alpha = 0.001,loss="hinge", penalty="l2") 0.74 0.97 0.91
                                                         #(loss="squared_hinge", penalty="none") 0.21 0.999 0.81
                                                         #(alpha = 0.01, loss="hinge", penalty="elasticnet") 0.6 0.99 0.89
                                                         #(alpha = 0.0001, loss = "perceptron", penalty = "elasticnet") 0.46 1.00 0.87
SGD.fit(train_x, train_y)
predicted_y = SGD.predict(test_x)
scores = cross_val_score(SGD, train_x, train_y, cv=10) #10-fold cross validation 
print("SGD 10-fold cross validation Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
cnf_matrix = confusion_matrix(test_y, predicted_y, labels = [1,0]) # Calculate confusion matrix 
np.set_printoptions(precision=2)
plt.figure()# Plot confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix for SGD')
metric = precision_recall_fscore_support(test_y, predicted_y, average='macro')
accuracy = metrics.accuracy_score(test_y, predicted_y)
print('SGD accuracy: {0}, SGD precision: {1}, SGD recall: {2}, SGD Fscore: {3}'.format(accuracy, metric[0], metric[1], metric[2]))

plt.show()

#--------------------------------------------------------------------------------------------------
# fine tune 
#--------------------------------------------------------------------------------------------------
# Set the parameters by cross-validation
# [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': [ 'none', 'l2', 'l1','elasticnet']},

tuned_parameters = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': [ 'none', 'l2', 'l1','elasticnet'], 
                    'alpha':[0.0001, 0.001, 0.01]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SGDClassifier(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(train_x, train_y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_y, clf.predict(test_x)
    print(classification_report(y_true, y_pred))
    print()
