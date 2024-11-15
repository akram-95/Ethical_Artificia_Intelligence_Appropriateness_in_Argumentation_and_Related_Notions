from django.http import JsonResponse
from rest_framework.decorators import api_view

from project.predict import classify_text


@api_view(['POST'])
def classify_text_view(request):
    # Extract the text from the POST request
    data = request.data
    text = data.get('text', None)

    # Validate input
    if not text:
        return JsonResponse({"error": "No text provided"}, status=400)

    # Get prediction from the classifier
    prediction = classify_text(text)

    # Send response
    return JsonResponse({
        "text": text,
        "predictions": prediction  # Return the first prediction result
    })
