from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict_crime, name='predict_crime'),
]
