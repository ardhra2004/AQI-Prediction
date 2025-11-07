from django.db import models

class AQIData(models.Model):
    city = models.CharField(max_length=100)
    date = models.DateField()
    pm25 = models.FloatField()
    pm10 = models.FloatField()
    no = models.FloatField()
    no2 = models.FloatField()
    nox = models.FloatField()
    nh3 = models.FloatField()
    co = models.FloatField()
    so2 = models.FloatField()
    o3 = models.FloatField()
    benzene = models.FloatField()
    toluene = models.FloatField()
    aqi = models.FloatField()
    aqi_bucket = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.city} ({self.date}) - {self.aqi_bucket}"
