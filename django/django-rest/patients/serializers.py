from datetime import date
from rest_framework import serializers
from .models import Patient, Insurance, MedicalRecord
from bookings.serializers import AppointmentSerializer

class PatientSerializer(serializers.ModelSerializer):

    appointments = AppointmentSerializer(many=True, read_only=True)
    age = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
        'id',
        'first_name',
        'last_name',
        'date_of_birth',
        'age',
        'contact_number',
        'email',
        'address',
        'medical_history',
        'appointments'
        ]

    def get_age(self, obj):
        age_td = date.today() - obj.date_of_birth
        return age_td.days // 365




class InsuranceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Insurance
        fields = '__all__'

class MedicalSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalRecord
        fields = '__all__'
