from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def sumary(request):
    name = request.data.get('name', 'Unknown')
    return Response({"message": f"Hello, {name}!"})