from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key_path = os.environ.get('GEMINI_API_KEY')

if not api_key_path:
    raise Exception("GEMINI_API_KEY environment variable is not set")


def process_receipt(image_file):
    # Save the file to a temporary location
    filename = secure_filename(image_file.filename)
    temp_path = os.path.join('/tmp', filename)
    image_file.save(temp_path)

    # Upload the image to Gemini
    uploaded_image = genai.upload_file(temp_path)

    # Create a Gemini model instance
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = """
    Analyze the following image of a grocery store receipt, which may be in Thai or English:

    [Image]

    Identify and extract the following information for each item:

    1. **Ingredient Name:** The name of the product or ingredient.
    2. **Price:** The price of the item.
    3. **Quantity:** The quantity purchased.

    Present the information in a structured JSON format, with each item represented as an object containing the three fields: `ingredientName`, `price`, and `quantity`.
    """

    # Generate text based on the image
    result = model.generate_content([uploaded_image, "\n\n", prompt])

    # Remove the temporary file after uploading
    os.remove(temp_path)

    # Convert the result text to JSON
    result_json = json.loads(result.text)

    # Return the JSON response
    return result_json


@app.route("/")
def hello_world():
    return "<p>Welcome to BakingUp Receipt Scanner</p>"


@app.route('/scan_receipt', methods=['GET'])
def scan_receipt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the uploaded file
    extracted_text = process_receipt(file)

    return extracted_text


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)
