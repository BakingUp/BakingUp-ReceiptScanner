from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import imutils
import cv2
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key_path = os.environ.get('GEMINI_API_KEY')

if not api_key_path:
    raise Exception("GEMINI_API_KEY environment variable is not set")


def orient_vertical(img):
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        rotated = imutils.rotate(img, angle=270)
    else:
        rotated = img.copy()

    return rotated

def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    return edged


def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    return dilated


def find_receipt_bounding_box(binary, img):
    global rect

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_cnt)
    box = np.intp(cv2.boxPoints(rect))
    boxed = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 20)
    return boxed, largest_cnt


def find_tilt_angle(largest_contour):
    angle = rect[2]  # Find the angle of vertical line
    print("Angle_0 = ", round(angle, 1))
    if angle < -45:
        angle += 90
        print("Angle_1:", round(angle, 1))
    else:
        uniform_angle = abs(angle)
    print("Uniform angle = ", round(uniform_angle, 1))
    return rect, uniform_angle


def adjust_tilt(img, angle):
    if angle >= 5 and angle < 80:
        rotated_angle = 0
    elif angle < 5:
        rotated_angle = angle
    else:
        rotated_angle = 270+angle
    tilt_adjusted = imutils.rotate(img, rotated_angle)
    delta = 360-rotated_angle
    return tilt_adjusted, delta


def crop(img, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]
    return cropped


def enhance_txt(img):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w*0.05)
    w2 = int(w*0.95)
    h1 = int(h*0.05)
    h2 = int(h*0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.98  # % of average brightness

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return binary


def preprocess_image(image_path):
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    rotated = orient_vertical(raw_rgb)
    edged = sharpen_edge(rotated)
    binary = binarize(edged, 100)

    try:
        # Try to find the bounding box
        boxed, largest_cnt = find_receipt_bounding_box(binary, rotated)
        rect, angle = find_tilt_angle(largest_cnt)
        tilted, delta = adjust_tilt(boxed, angle)

        # Check if cropping is needed
        x, y, w, h = cv2.boundingRect(largest_cnt)
        img_height, img_width = tilted.shape[:2]

        if w > img_width * 0.5 and h < img_height * 0.5:
            # Crop if the bounding box is significantly smaller than the image
            cropped = crop(tilted, largest_cnt)
        else:
            # Skip cropping and use the rotated image
            cropped = rotated
    except Exception as e:
        print(f"Error in find_receipt_bounding_box: {e}")
        # Skip bounding box and use the rotated image directly
        cropped = rotated

    # Enhance text in the cropped or rotated image
    enhanced = enhance_txt(cropped)

    # Save the enhanced image to a temporary file
    filename = secure_filename(os.path.basename(image_path + "_enhanced.jpg"))
    temp_path = os.path.join('/tmp', filename)
    cv2.imwrite(temp_path, enhanced)
    return temp_path


def process_receipt(image_file):
    # Save the file to a temporary location
    filename = secure_filename(image_file.filename)
    temp_path = os.path.join('/tmp', filename)
    image_file.save(temp_path)

    # Preprocess the image
    temp_path = preprocess_image(temp_path)

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

    **Note:** The receipt image has been preprocessed to improve readability, including enhancing text contrast and reducing noise. Any faded text or low-contrast areas are likely adjusted for better clarity.

    **Focus on identifying and extracting information for items that are likely ingredients.** For each ingredient:

    1. **Ingredient Name:** The name of the product or ingredient.
    2. **Price:** The price of the item.
    3. **Quantity:** The quantity purchased.

    **Exclude non-ingredient items** like taxes, subtotal, total, store name, date, etc.

    Present the information in a structured JSON format, with each ingredient represented as an object containing the three fields: `ingredientName`, `price`, and `quantity`.
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
