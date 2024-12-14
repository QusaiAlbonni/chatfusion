from chatfusion import configure
from chatfusion import GeneratorFactory
from chatfusion import Prompt
import os
from dotenv import load_dotenv

load_dotenv()

configure(openai_api_key=os.environ.get('KEY'), gemini_api_key=os.environ.get('G_KEY'))

generator = GeneratorFactory().create_generator(model_name='gemini-1.5-pro')

prompt = Prompt().chat().system("You are a helpful assistent called Emillie")

print("Emillie: Hello, how can I be of assistence?\n")

while True:
    user_input = input("You: ")
    print("\n")
    
    prompt = prompt.user(user_input)
    
    response = generator.generate_response(prompt, retry=True)
    text = response.get_text()
        
    print("Emillie:", text)

    prompt = prompt.assistant(text)