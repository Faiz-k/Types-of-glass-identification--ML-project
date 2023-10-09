from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox

def home(request):
    if request.method == "POST":
        try:
            ri = float(request.POST.get('ri'))
            na = float(request.POST.get('na'))
            mg = float(request.POST.get('mg'))
            al = float(request.POST.get('al'))
            si = float(request.POST.get('si'))
            k = float(request.POST.get('k'))
            ca = float(request.POST.get('ca'))
            ba = float(request.POST.get('ba'))
            fe = float(request.POST.get('fe'))
        except ValueError:
            return render(request, 'home.html', context={'error': 'Invalid input. Please enter numeric values.'})

        # Load and preprocess the dataset (similar to your code)
        # ...
        path="C:\\Users\\mf879\OneDrive\\Desktop\\45_Glassidentification\\glass.csv"
        data = pd.read_csv(path)
        X = data.drop('Type', axis=1)
        y = data['Type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        skewed_features = ['Na', 'Mg', 'K', 'Ba', 'Fe']
        for feature in skewed_features:
            X_train[feature] = boxcox(X_train[feature] + 1)[0]
            X_test[feature] = boxcox(X_test[feature] + 1)[0]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)
        # Use the Naive Bayes classifier to predict the glass type
        y = nb_classifier.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
        return render(request, "home.html", context={'predicted_type': y[0]})

    return render(request, 'home.html')


