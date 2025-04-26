from flask import Flask, jsonify, request
from google.cloud import vision
import subprocess
import os
import cv2
import requests
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import io
import base64
import json
import time
# pillow library for image

# Load environment variables
load_dotenv()

# GPT API Key
gpt_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GPT_GITHUB_KEY"]
)

hf_client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HUGGING_FACE_TOKEN"],
)

app = Flask(__name__)

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Load the valid food names from a file
def load_food_names():
    with open("food_names.txt", "r") as file:
        return set(line.strip().lower() for line in file)

valid_food_names = load_food_names()

def capture_image():
    """Captures an image from the camera."""
    image_path = "image.jpg"
    subprocess.run(["libcamera-jpeg", "-o", image_path])
    return image_path

def detect_labels(image_path):
    """Detects objects and returns the highest-scoring valid food label."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects_response = client.object_localization(image=image)
    localized_objects = objects_response.localized_object_annotations

    original_image = cv2.imread(image_path)
    height, width, _ = original_image.shape

    results = []

    for obj in localized_objects:
        vertices = obj.bounding_poly.normalized_vertices
        x_min = int(vertices[0].x * width)
        y_min = int(vertices[0].y * height)
        x_max = int(vertices[2].x * width)
        y_max = int(vertices[2].y * height)

        cropped_image = original_image[y_min:y_max, x_min:x_max]
        _, cropped_image_buffer = cv2.imencode('.jpg', cropped_image)
        cropped_image_content = cropped_image_buffer.tobytes()

        cropped_image_vision = vision.Image(content=cropped_image_content)
        labels_response = client.label_detection(image=cropped_image_vision)
        labels = labels_response.label_annotations

        valid_labels = [label for label in labels if label.description.lower() in valid_food_names]
        if valid_labels:
            best_label = max(valid_labels, key=lambda label: label.score)
            results.append(best_label.description)

    return results

@app.route("/food_recognition")
def food_recognition():
    try:
        image_path = capture_image()
        objects_with_best_label = detect_labels(image_path)

        return jsonify({
            "status": "success",
            "objects": objects_with_best_label
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route("/recipe-suggestion", methods=["POST"])
def recipe_suggestion():
    try:
        ingredients = request.json.get("ingredients", [])
        ingredients = list(set(i.strip().lower() for i in ingredients if i.strip()))

        prompt = f"""
        Generate an Arab recipe using these available ingredients in the fridge: {', '.join(ingredients)} (not necessarily all of them + return the names of the ingredients only).
        Return a valid JSON object in this exact format:
        {{
          "recipe_name": "A name for the suggested dish",
          "recipe_image": "a description of the dish for image generation"
          "ingredients": {{
            "available": ["List", "of", "ingredients", "present"],
            "missing": ["List", "of", "ingredients needed for the recipe", " but not", "available in our fridge"]
          }},
          "instructions": ["list of Step-by-step cooking instructions"]
        }}
        Make sure the response is ONLY a JSON object (no explanations or comments).
        """

        # Step 1: Get the recipe from GPT
        response = gpt_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful recipe assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o",
            temperature=1,
            max_tokens=4096,
            top_p=1
        )

        print(response.choices[0].message.content)
        recipe_text = response.choices[0].message.content.strip()
        recipe_json = json.loads(recipe_text)  # Parse GPT JSON safely

        # Step 2: Use recipe_name to generate an image with FLUX
        recipe_placeholder = recipe_json["recipe_image"]
        image = hf_client.text_to_image(recipe_placeholder, model="black-forest-labs/FLUX.1-schnell") # output is a PIL.Image object

        # Step 3: Save to Desktop and convert to base64
        image_path = "/home/JSL/Desktop/FridgeX/recipe_image.png"  
        image.save(image_path, format='PNG') # save for testing
        image_io = io.BytesIO()
        image.save(image_io, format='PNG')
        image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8') # convert to base64

        print(recipe_json)

        # Step 4: Return full response with base64 image instead of just URL
        return ({
            "status": 200,
            "recipe": recipe_json,
            "recipe_image": image_base64,
        })

        #return jsonify({
        #    "status": "success",
        #    "recipe": recipe_text
        #})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route("/inside-view")
def inside_view():
    try:
        # Step 1: Capture a fresh image
        image_path = capture_image()
        time.sleep(1.5)
        
        # Step 2: Read the captured image and encode it to base64
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Step 3: Return the base64 image
        return jsonify({
            "status": "success",
            "image": image_base64
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

        

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
