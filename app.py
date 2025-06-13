from flask import Flask, jsonify, request
from google.cloud import vision
import subprocess
import os
import cv2
import io
import base64
import json
import time
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
from huggingface_hub import InferenceClient

from door_monitor import start_door_monitoring, get_door_status
from door_closing import close_door
from door_lock import lock_door, unlock_door

door_is_locked = False
# Load environment variables
load_dotenv()

# Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Clients
gpt_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GPT_GITHUB_KEY"],
)

hf_client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HUGGING_FACE_TOKEN"],
)

app = Flask(__name__)
CORS(app)

# Load valid food names
def load_food_names():
    with open("food_names.txt", "r") as file:
        return set(line.strip().lower() for line in file)

valid_food_names = load_food_names()

def capture_image():
    image_path = "image.jpg"
    subprocess.run(["libcamera-jpeg", "-o", image_path])
    return image_path

def detect_labels(image_path):
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
        _, buffer = cv2.imencode('.jpg', cropped_image)
        cropped_image_content = buffer.tobytes()

        label_resp = client.label_detection(image=vision.Image(content=cropped_image_content))
        valid_labels = [label for label in label_resp.label_annotations if label.description.lower() in valid_food_names]
        if valid_labels:
            best_label = max(valid_labels, key=lambda l: l.score)
            results.append(best_label.description)

    return results

@app.route("/food-recognition")
def food_recognition():
    try:
        image_path = capture_image()
        detected_items = detect_labels(image_path)
        return jsonify({"status": "success", "objects": detected_items})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
          "time": "an integer that represents the preparation time in minutes",
          "recipe_image": "a description of the dish for image generation",
          "ingredients": {{
            "available": ["List", "of", "ingredients", "present"],
            "missing": ["List", "of", "ingredients needed for the recipe", " but not", "available in our fridge"]
          }},
          "instructions": ["list of Step-by-step cooking instructions"]
        }}
        Make sure the response is ONLY a JSON object (no explanations or comments).
        """

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

        recipe_json = json.loads(response.choices[0].message.content.strip())

        image = hf_client.text_to_image(recipe_json["recipe_image"], model="black-forest-labs/FLUX.1-schnell")
        image_io = io.BytesIO()
        image.save(image_io, format='PNG')
        image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')

        return {
            "status": 200,
            "recipe": recipe_json,
            "recipe_image": image_base64,
        }
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/inside-view")
def inside_view():
    try:
        image_path = capture_image()
        time.sleep(1.5)
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({"status": "success", "image": base64_img})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/door-status")
def door_status_api():
    return jsonify({"status": "success", "door_open": get_door_status()})

@app.route("/api/door-open-alert", methods=["POST"])
def door_open_alert():
    print("[API] Door open alert received. Send push notification to app here.")
    return jsonify({"status": "received"}), 200

@app.route("/close-door", methods=["POST"])
def close_door_api():
    try:
        close_door()
        return jsonify({"status": "success", "message": "Door closed."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route('/door-lock-status', methods=['GET'])
def door_lock_status():
    return jsonify({"locked": door_is_locked})
        
@app.route("/lock-door", methods=["POST"])
def lock_door_api():
    global door_is_locked
    try:
        lock_door()
        door_is_locked = True
        return jsonify({"status": "success", "message": "Door locked."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/unlock-door", methods=["POST"])
def unlock_door_api():
    global door_is_locked
    try:
        unlock_door()
        door_is_locked = False
        return jsonify({"status": "success", "message": "Door unlocked."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route("/register-device", methods=["POST"])
def register_device():
    try:
        data = request.get_json()
        fcm_token = data.get("fcm_token")

        if not fcm_token:
            return jsonify({"status": "error", "message": "Missing FCM token"}), 400

        # Save to a local file (or replace with DB logic)
        with open("fcm_token.txt", "w") as file:
            file.write(fcm_token)

        print(f"[TOKEN] Registered new device token: {fcm_token}")
        return jsonify({"status": "success", "message": "FCM token registered."}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    start_door_monitoring()
    app.run(host="0.0.0.0", port=5000)
