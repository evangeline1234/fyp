# user/urls.py
from django.urls import path
from .views import PredictAvailabilityView

urlpatterns = [
    path('predict/', PredictAvailabilityView.as_view(), name='predict-availability'),
]
