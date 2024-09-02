from django.test import TestCase
from django.urls import reverse


# Create your tests here.
class ProductListViewTests(TestCase):

    def test_should_return_200(self):
        url = reverse("list_product")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
