import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set maximum length for generated responses
max_response_length = 50

def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response using the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_response_length, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and return it as a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    print("Chatbot: Hello! You can start chatting with me. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Generate the chatbot's response
        bot_response = generate_response(user_input)

        # Print the chatbot's response
        print("Chatbot:", bot_response)

if __name__ == "__main__":
    main()
