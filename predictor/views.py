from django.shortcuts import render
import pickle
import os
import numpy as np

# Load the saved ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

def predict_crime(request):
    context = {}
    if request.method == 'POST':
        try:
            population = float(request.POST.get('population', 0))
            unemployment = float(request.POST.get('unemployment', 0))
            education = float(request.POST.get('education', 0))

            # Predict crime rate
            features = np.array([[population, unemployment, education]])
            prediction = model.predict(features)[0]
            context['prediction'] = round(prediction, 2)
        except Exception as e:
            context['error'] = str(e)

    return render(request, 'predictor/index.html', context)
