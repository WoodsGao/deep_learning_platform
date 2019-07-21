from .models import Dataset, Classification, ImageData, ValidateImageTemp
from rest_framework import serializers


class ValidateImageTempSerializer(serializers.ModelSerializer):
    class Meta:
        model = ValidateImageTemp
        fields = '__all__'


class ImageDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageData
        fields = '__all__'

    classification_name = serializers.SerializerMethodField()
    dataset_name = serializers.SerializerMethodField()

    def get_dataset_name(self, obj):
        return obj.classification.dataset.name

    def get_classification_name(self, obj):
        return obj.classification.name


class ClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Classification
        fields = '__all__'

    image_urls = serializers.SerializerMethodField()
    dataset_name = serializers.SerializerMethodField()
    image_counts = serializers.SerializerMethodField()

    def get_image_urls(self, obj):
        images = ImageData.objects.filter(classification_id=obj.id)
        return [i.img.url for i in images]

    def get_dataset_name(self, obj):
        return obj.dataset.name

    def get_image_counts(self, obj):
        all_images = ImageData.objects.filter(classification_id=obj.id)
        return all_images.count()


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'

    classifications = serializers.SerializerMethodField()
    image_counts = serializers.SerializerMethodField()

    def get_classifications(self, obj):
        all_classifications = Classification.objects.filter(dataset_id=obj.id)
        classifications_serializer = ClassificationSerializer(all_classifications, many=True)
        return classifications_serializer.data

    def get_image_counts(self, obj):
        all_images = ImageData.objects.filter(classification__dataset_id=obj.id)
        return all_images.count()
