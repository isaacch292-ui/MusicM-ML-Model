#import required libaries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(r'C:\Users\Castle\Desktop\musicM\archive\Data\features_3_sec.csv')

# Remove 'filename' column
if 'filename' in data.columns:
    data = data.drop(['filename'],axis=1)

#split the dataset into features (x) and labels (y)
x=data.drop("label",axis=1)
y=data["label"]   

#split data into training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#normalize feature values using standardscaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#train the random forest classifier
clf=RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

#predict genres for test data and evaluate model performance
y_pred=clf.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\n Classification Report: \n", classification_report(y_test, y_pred))


#predict genres for the first 5 songs in the dataset
new_songs=x.iloc[:5]
new_songs_scaled= scaler.transform(new_songs)
predictions = clf.predict(new_songs_scaled)


#display predicted genres for the selected songs
for i, genre in enumerate(predictions, start=1):
   print(f"song {i} predicted genre: {genre} ")


import joblib
joblib.dump(clf, 'model.pkl')

   