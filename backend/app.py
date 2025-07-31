from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows requests from React Native (cross-origin)

@app.route('/generate_beta', methods=['POST'])
def generate_beta():
    data = request.get_json()

    # Optional: print for debugging
    print("Received JSON:", data)

    # Validate basic structure
    if not all(k in data for k in ("problem_name", "wall", "climber", "holds")):
        return jsonify({"error": "Invalid payload structure"}), 400

    try:
        result = run_beta_generation(data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
