from flask import Flask, request, jsonify
from chat import initialize_vector_store, get_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize vector DB on startup
initialize_vector_store()

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        answer = get_response(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)