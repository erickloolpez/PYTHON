from .serializers import PatientSerializer
from .models import Patient

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, CreateAPIView, RetrieveUpdateDestroyAPIView



# Create your views here.

# GET /api/patients -> Listar
# POST /api/patienst -> Crear
# PUT /api/patients/<pk> -> Modificacion
# GET /api/patients/<pk> -> Detalle
# DELETE /api/patients/<pk> -> Eliminar


class ListPatientsView(ListAPIView, CreateAPIView):
    """
    Obtiene la lista de pacientes.
    """
    allowed_methods = ["GET", "POST"]
    serializer_class = PatientSerializer
    queryset = Patient.objects.all()


class DetailPatientView(RetrieveUpdateDestroyAPIView):
    allowed_methods = ["GET","PUT","DELETE"]
    serializer_class = PatientSerializer
    queryset = Patient.objects.all()



# Forma antigua de como haciamos el enrutamiento
@api_view(["GET", "POST"])
def list_patients(request):
    if request.method == "GET":
        patients = Patient.objects.all()
        serializer = PatientSerializer(patients, many=True)
        return Response(serializer.data)

    if request.method == "POST":
        serializer = PatientSerializer(data=request.data)
        serializer.is_valid(
            raise_exception=True
        )  # Esto es para devolver algo en vez de que todo crashee
        serializer.save()
        return Response(status=status.HTTP_201_CREATED)


@api_view(["GET", "PUT", "DELETE"])
def detail_patient(request, pk):
    try:
        patient = Patient.objects.get(id=pk)
    except Patient.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        serializer = PatientSerializer(patient)
        return Response(serializer.data)

    if request.method == "PUT":
        serializer = PatientSerializer(patient, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    if request.method == "DELETE":
        patient.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
