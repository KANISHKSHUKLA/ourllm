from flask import Flask, request, jsonify
from inference import generate_text

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    generated_text = generate_text(prompt)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
