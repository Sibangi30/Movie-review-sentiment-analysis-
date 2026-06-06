accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for name,clf in clfs.items():

    # Correctly unpack all four values returned by train_classifier
    current_accuracy, current_precision, current_recall, current_f1 = train_classifier(clf, X_train,y_train,X_test,y_test)

    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("F1 score -",current_f1)
    print("Recall - ",current_recall)
    print("\n")

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)
    f1_scores.append(current_f1)
    from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative','Positive'])
disp.plot(cmap='Blues')
plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train,cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())

# Precision, Recall, F1 (binary classification)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Detailed per-class report
print("\nClassification Report:\n", classification_report(y_true, y_pred))
