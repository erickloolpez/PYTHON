from django.contrib import admin
from .models import Order, OrderProduct


# Register your models here.
class OrderProductInlineAdmin(admin.TabularInline):
    model = OrderProduct
    extra = 0


class OrderAdmin(admin.ModelAdmin):
    model = Order
    inlines = [OrderProductInlineAdmin]


admin.site.register(Order, OrderAdmin)