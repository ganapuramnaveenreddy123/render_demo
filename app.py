from flask import Flask, request, jsonify
from rag_model import get_answer

app = Flask(__name__)

@app.route("/")
def home():
    return "RAG Chatbot Running 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")

    answer = get_answer(question)

    return jsonify({
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=5000)