from django.urls import path
from .views import *
urlpatterns = [
    path('test/', test, name='test'),
    path('train_model/', train_model, name='train_model'),
    path('list_tickers/', list_tickers, name='list_tickers'),
]