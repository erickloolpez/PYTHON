from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import ListPatientsView, DetailPatientView
from .viewsets import PatientViewSet


router = DefaultRouter()
router.register('patients', PatientViewSet)

urlpatterns = [
    # path("patients/", ListPatientsView.as_view()), Esto ya no es asi porque ya tenemos el viewsets
    path("patients/<int:pk>/", DetailPatientView.as_view()),
]+ router.urls
