from chatfusion import configure
from chatfusion import GeneratorFactory
from chatfusion import Prompt

configure(gemini_api_key="YOUR_API_KEY")
configure(openai_api_key="YOUR_API_KEY")

generator = GeneratorFactory().create_generator(model_name='gemini-1.5-pro-latest')

prompt = Prompt().chat().system("You are a helpful assistent named John.")

print("John: Hello, how can I be of assistence?\n")

while True:
    user_input = input("You: ")
    print("\n")
    
    prompt = prompt.user(user_input)
    
    response = generator.generate_response(prompt, retry=True)
    text = response.text()
        
    print("John:", text)

    prompt = prompt.assistant(text)