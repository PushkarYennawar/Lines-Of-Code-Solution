from django.db import models

# Create your models here.
class Image(models.Model):
 photo = models.ImageField(upload_to="myimage")
 date = models.DateTimeField(auto_now_add=True)
 classification = models.CharField(max_length=10, blank=True)