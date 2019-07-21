from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from .models import Dataset, Classification, ImageData, ValidateImageTemp
from .serializers import DatasetSerializer, ClassificationSerializer, ImageDataSerializer, ValidateImageTempSerializer
from .src import create_cnn_model, validate_image, resize_image


# Create your views here.

class DatasetViewSet(ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

    def update(self, request, *args, **kwargs):
        dataset = Dataset.objects.get(pk=kwargs.get('pk'))
        if dataset.target_epoches != 0:
            target_epoches = int(request.data.get('target_epoches'))
            dataset.target_epoches = target_epoches
            dataset.save()
            response = super(DatasetViewSet, self).retrieve(request, *args, **kwargs)
            return response
        response = super(DatasetViewSet, self).update(request, *args, **kwargs)
        return response

    def destroy(self, request, *args, **kwargs):
        dataset = Dataset.objects.get(pk=kwargs.get('pk'))
        if dataset.target_epoches != 0:
            return Response({'message': 'This dataset is under training.'})
        response = super(DatasetViewSet, self).destroy(request, *args, **kwargs)
        return response


class ClassificationViewSet(ModelViewSet):
    queryset = Classification.objects.all()
    serializer_class = ClassificationSerializer

    def create(self, request, *args, **kwargs):
        dataset_id = request.data.get('dataset')
        dataset = Dataset.objects.get(id=dataset_id)
        if dataset.target_epoches != 0:
            return Response({'message': 'This dataset is under training.'})
        response = super(ClassificationViewSet, self).create(request, *args, **kwargs)
        create_cnn_model(dataset_id)
        return response

    def update(self, request, *args, **kwargs):
        dataset_id = request.data.get('dataset')
        dataset = Dataset.objects.get(id=dataset_id)
        if dataset.target_epoches != 0:
            return Response({'message': 'This dataset is under training.'})
        response = super(ClassificationViewSet, self).update(request, *args, **kwargs)
        create_cnn_model(dataset_id)
        return response

    def destroy(self, request, *args, **kwargs):
        classification = Classification.objects.get(pk=kwargs.get('pk'))
        dataset_id = classification.dataset.id
        dataset = Dataset.objects.get(id=dataset_id)
        if dataset.target_epoches != 0:
            return Response({'message': 'This dataset is under training.'})
        response = super(ClassificationViewSet, self).destroy(request, *args, **kwargs)
        create_cnn_model(dataset_id)
        return response


class ImageDataViewSet(ModelViewSet):
    queryset = ImageData.objects.all()
    serializer_class = ImageDataSerializer

    def create(self, request, *args, **kwargs):
        response = super(ImageDataViewSet, self).create(request, *args, **kwargs)
        image_id = response.data.get('id')
        image_data = ImageData.objects.get(id=image_id)
        resize_image(image_data.img.path)
        return response

    def update(self, request, *args, **kwargs):
        image = ImageData.objects.get(pk=kwargs.get('pk'))
        classification = Classification.objects.get(id=request.data.get('classification'))
        image.classification = classification
        image.save()
        response = super(ImageDataViewSet, self).retrieve(request, *args, **kwargs)
        return response


class ValidateImageTempViewSet(ModelViewSet):
    queryset = ValidateImageTemp.objects.all()
    serializer_class = ValidateImageTempSerializer

    def create(self, request, *args, **kwargs):
        response = super(ValidateImageTempViewSet, self).create(request, *args, **kwargs)
        temp_id = response.data.get('id')
        temp_image = ValidateImageTemp.objects.get(id=temp_id)
        resize_image(temp_image.img.path)
        response = validate_image(temp_image)
        # if not temp_image.save_to_db:
        temp_image.delete()
        return Response(response)
