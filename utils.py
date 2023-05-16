

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def apply_svm_model(X, y, kernel=['linear'], C=[1.0], gamma=['scale']):
    accs = []
    f1_accs = []

    for c in C:
        for g in gamma:
            for k in kernel:
                # Set up k-fold cross-validation with 10 folds
                kf = RepeatedStratifiedKFold(n_splits=10,n_repeats=1,random_state=42)

                # Initialize an empty list to store the cross-validation scores
                f1_scores = []
                accurcy_score=[]
                cms = []
                i =0
                for train_index, test_index in kf.split(X, y):
                    if type(X) == np.ndarray:
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                    else:
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    smote = SMOTE(random_state=42)
                    
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    if kernel == 'poly':
                        model = SVC(gamma=g, random_state=42, kernel=k, C=c, degree=3)
                    else:
                        model = SVC(gamma=g, random_state=42, kernel=k, C=c)

                    model.fit(X_train, y_train)

                    # Make predictions on the testing data
                    y_pred = model.predict(X_test)
                    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                    accurcy_score.append(accuracy_score(y_test, y_pred))
                    
                    # plot the confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    cms.append(cm)
                    
                    i+=1

                # plot all the confusion matrices
                fig, axes = plt.subplots(2, 5, figsize=(20, 10))
                axes = axes.flatten()
                for i, cm in enumerate(cms):
                    sns.heatmap(cm, annot=True, ax=axes[i], cmap='Blues', fmt='g')
                    axes[i].set_title('Confusion Matrix {}'.format(i+1))
                    axes[i].set_xlabel('Predicted label')
                    axes[i].set_ylabel('True label')
                    plt.tight_layout()
                plt.savefig('outputs_svm/confusion_matrix_'+str(k)+'_'+str(c)+'_'+str(g)+'.png', dpi=300, bbox_inches='tight')

                # don't show the plots
                plt.close()
                
                f1_scores=np.array(f1_scores)
                accurcy_score=np.array(accurcy_score)
                # Compute the average accuracy score over all folds
                accs.append(accurcy_score.mean())
                f1_accs.append(f1_scores.mean())
                print ("........ for kernel: ", k, "gamma: ",g," and C: ", c, ".......")
                print(f"- Average weighted F1 score {f1_scores.mean():.5f} (+/- {f1_scores.std():.5f})")
                print(f"- Average Accuracy score {accurcy_score.mean():.5f} (+/- {accurcy_score.std():.5f})")
                print(".................................................................")

    

    return kernel, C, gamma, accs, f1_accs