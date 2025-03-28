from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    carparkcode = serializers.CharField(max_length=50)
    datetime = serializers.DateTimeField()