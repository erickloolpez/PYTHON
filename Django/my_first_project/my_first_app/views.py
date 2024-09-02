from django.views.generic.base import TemplateView
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def my_view(request, *args, **kwargs):
    print(args)
    print(kwargs)
    car_list=[
        {"title": "BMW"},
        {"title":"Mazda"}
    ]

    context = {
        "car_list":car_list
    }
    return render(request, "my_first_app/car_list.html",context)

class CarListView(TemplateView):
    template_name="my_first_app/car_list.html"

    def get_context_data(self):
        car_list=[
            {"title": "BMW"},
            {"title":"Mazda"}
        ]

        return {
            "car_list":car_list
        }


def my_test_view(request, *args, **kwargs):
    print(args)
    print(kwargs)
    return HttpResponse('')