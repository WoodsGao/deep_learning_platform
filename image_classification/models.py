from django.db import models


# Create your models here.

class Dataset(models.Model):
    name = models.CharField(max_length=100, unique=True)
    acc = models.FloatField(default=0)
    loss = models.FloatField(default=1000)
    target_epoches = models.IntegerField(default=0)
    now_epoches = models.IntegerField(default=0)
    lr = models.FloatField(default=0)

    def __str__(self):
        return self.name


class Classification(models.Model):
    name = models.CharField(max_length=100)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)

    def __str__(self):
        return '|'.join([self.dataset.name, self.name])


class ImageData(models.Model):
    img = models.ImageField(upload_to='image_classification/')
    classification = models.ForeignKey(Classification, on_delete=models.CASCADE)

    def __str__(self):
        return '|'.join([self.dataset.name, self.classification.name])


class ValidateImageTemp(models.Model):
    img = models.ImageField(upload_to='image_classification/')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    save_to_db = models.BooleanField(default=False)
