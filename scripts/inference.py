import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from a local directory
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")
    
    try:
        print(f"Loading model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_text(prompt, model, tokenizer, max_length=100):
    """
    Generate text based on a prompt
    """
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate text
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise

def main():
    # Define the path to your saved model
    model_path = os.path.join(os.getcwd(), "saved_model")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Example prompt
        prompt = input("Enter your prompt: ")
        
        # Generate text
        generated_text = generate_text(prompt, model, tokenizer)
        
        print("\nGenerated Text:")
        print(generated_text)
        
    except FileNotFoundError:
        print(f"\nError: Model not found at {model_path}")
        print("Make sure you have trained the model and saved it in the 'saved_model' directory")
        print("The directory structure should be:")
        print("our-llm/")
        print("└── saved_model/")
        print("    ├── config.json")
        print("    ├── pytorch_model.bin")
        print("    └── tokenizer_config.json")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()