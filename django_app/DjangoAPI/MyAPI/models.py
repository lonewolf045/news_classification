from django.db import models

# Create your models here.

class news(models.Model):
    number=models.IntegerField(default=0)
    headline=models.CharField(max_length=100)
    name=models.CharField(max_length=20)
    date=models.CharField(max_length=10)
    content=models.CharField(max_length=1000)

    def __str__(self):
        return self.number
