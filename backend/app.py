from flask import Flask, request, jsonify, send_from_directory
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
        image_file = request.files['image']
        filename = f"uploads/{image_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        image_file.save(filename)

        payload_json = request.form['data']
        data = json.loads(payload_json)
        data["image_path"] = filename

        result = run_beta_generation(data)  # Assume it returns path like 'outputs/RedV5_slideshow.pdf'
        return jsonify({
            "status": "success",
            "result": result,
            "pdf_url": f"/download_pdf/{os.path.basename(result)}"
        })
    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/download_pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    return send_from_directory('outputs', filename, as_attachment=True)
