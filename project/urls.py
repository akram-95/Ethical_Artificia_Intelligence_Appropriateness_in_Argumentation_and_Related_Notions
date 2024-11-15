from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_text_view, name='classify_text'),
]
