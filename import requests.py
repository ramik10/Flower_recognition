import requests

# Replace with the URL of your deployed Flask application
url = 'http://localhost:5000/predict'

# Replace with the path to an image file you want to test
image_path = 'C:/Users/ramik/Desktop/21652746_cc379e0eea_m.jpg'

# Send a POST request to the API
response = requests.post(url, files={'image': open(image_path, 'rb')})

# Print the response
print(response.json())
