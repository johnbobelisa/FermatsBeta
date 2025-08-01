from flask import Flask, request, jsonify
from flask_cors import CORS
from main import run_beta_generation
import os
import json

app = Flask(__name__)
CORS(app)

@app.route('/generate_beta', methods=['POST'])
def generate_beta():
    if 'image' not in request.files or 'data' not in request.form:
        return jsonify({"error": "Missing image or data in request"}), 400

    try:
        # 1. Save uploaded image
        image_file = request.files['image']
        filename = f"uploads/{image_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        image_file.save(filename)

        # 2. Parse JSON payload
        payload_json = request.form['data']
        data = json.loads(payload_json)
        data["image_path"] = filename   

        # 3. Run generation logic
        result = run_beta_generation(data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500
