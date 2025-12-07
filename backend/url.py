from django.urls import path
from .views import *
from .dataset_views import sp500_gics_sectors, gics_sector_returns
urlpatterns = [
    path('test/', test, name='test'),
    path('train_model/', train_model, name='train_model'),
    path('list_tickers/', list_tickers, name='list_tickers'),
    path('train_by_sectors/', train_by_sectors, name='train_by_sectors'),
    path('sp500/gics-sectors/', sp500_gics_sectors, name='sp500_gics_sectors'),
    path('sp500/gics-returns/', gics_sector_returns, name='gics_sector_returns'),
]