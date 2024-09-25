from rest_framework import viewsets, status 
from rest_framework.decorators import action
from rest_framework.response import Response
from .serializers import DoctorSerializer
from .models import Doctor

from rest_framework.permissions import IsAuthenticated#con esto solo ve la informacion pero nada mas.
from .permissions import IsDoctor
from bookings.serializers import AppointmentSerializer
from bookings.models import Appointment


class DoctorViewSet(viewsets.ModelViewSet):
    serializer_class = DoctorSerializer
    queryset = Doctor.objects.all()
    permission_classes = [IsAuthenticated, IsDoctor]

    @action(['POST'], detail=True, url_path='set-on-vacation')
    def set_on_vacation(self, requests, pk):
        doctor = self.get_object() # este metodo reemplaza el: self.queryset.filter(id=pk)
        doctor.is_on_vacation = True
        doctor.save()
        return Response({"status": "El doctor está en vacaciones"})

    @action(['POST'], detail=True, url_path='set-off-vacation')
    def set_off_vacation(self, requests, pk):
        doctor = self.get_object()
        doctor.is_on_vacation = False
        doctor.save()
        return Response({"status": "El doctor NO está en vacaciones"})

    @action(['POST','GET'],detail=True, serializer_class=AppointmentSerializer)
    def appointments(self, request, pk):
        doctor = self.get_object()
        data = request.data.copy()
        data['doctor'] = doctor.id

        if request.method == "POST":
            serializer = AppointmentSerializer(data=data)
            serializer.is_valid(raise_exception=True)
            serializer.save()

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        if request.method == "GET":
            appointments = Appointment.objects.filter(doctor=doctor)
            serializer = AppointmentSerializer(appointments, many=True)
            return Response(serializer.data)