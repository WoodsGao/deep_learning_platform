from django.urls import path, re_path
from . import views

urlpatterns = [
    path('dataset/', views.DatasetViewSet.as_view({'get': 'list', 'post': 'create'})),
    re_path(r'dataset/(?P<pk>\d+)/$',
            views.DatasetViewSet.as_view({'get': 'retrieve', 'put': 'update', 'delete': 'destroy'})),
    path('classification/', views.ClassificationViewSet.as_view({'get': 'list', 'post': 'create'})),
    re_path(r'classification/(?P<pk>\d+)/$',
            views.ClassificationViewSet.as_view({'get': 'retrieve', 'put': 'update', 'delete': 'destroy'})),
    path('image_data/', views.ImageDataViewSet.as_view({'get': 'list', 'post': 'create'})),
    re_path(r'image_data/(?P<pk>\d+)/$',
            views.ImageDataViewSet.as_view({'get': 'retrieve', 'put': 'update', 'delete': 'destroy'})),
    path('validate/', views.ValidateImageTempViewSet.as_view({'post': 'create'})),
]
