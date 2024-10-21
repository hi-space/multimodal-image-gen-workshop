from io import BytesIO
from PIL import Image
import base64
import requests
from IPython.display import display, HTML


# Function to encode image from bytes or PIL.Image
def encode_image_base64(image, format="WEBP", max_size=(1000, 1000)):
    # If the input is not an instance of PIL.Image, open it
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Resize the image
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Save the image to buffer and encode as base64
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=format)
    encoded_image = base64.b64encode(buffer.getvalue())
    return encoded_image.decode('utf-8')

# Function to encode image from a URL
def encode_image_base64_from_url(img_url, format="WEBP", max_size=(1000, 1000)):
    try:
        response = requests.get(img_url)
        response.raise_for_status()  # Raise an error for bad responses
        return encode_image_base64(BytesIO(response.content), format, max_size)
    except Exception as e:
        print(f"Error fetching image from URL: {e}")
        return None

# Function to encode image from a file path
def encode_image_base64_from_file(file_path, format="WEBP", max_size=(1000, 1000)):
    try:
        with open(file_path, 'rb') as img_file:
            image_data = img_file.read()
        return encode_image_base64(BytesIO(image_data), format, max_size)
    except Exception as e:
        print(f"Error reading image from file: {e}")
        return None

# Display image given base64-encoded string
def display_image(utf8_encoded_image, width=100):
    html = f'<img src="data:image/webp;base64,{utf8_encoded_image}" width="{width}"/>'
    display(HTML(html))
