from django.urls import path
from .views import sumary

urlpatterns = [
    path('sumary/', sumary),
]