from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_NAME = "microsoft/DialoGPT-medium"  # small/medium/large options

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# We'll keep simple chat history per session in the browser (sent to server)
@app.route("/")
def index():
    return render_template("index_local.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        user_msg = request.json.get("message", "")
        conversation_ids = request.json.get("history_ids", [])  # optional list of input_ids to continue
        # Build input
        new_user_input_ids = tokenizer.encode(user_msg + tokenizer.eos_token, return_tensors='pt').to(device)
        # If you want to carry the whole conversation, you can concatenate previous chat history tokens.
        bot_input_ids = new_user_input_ids
        if conversation_ids:
            # conversation_ids expected as a list of ints representing last input ids; easier approach below
            pass

        # generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1
        )
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

