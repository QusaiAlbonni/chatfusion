from chatfusion import configure
from chatfusion import GeneratorFactory
from chatfusion import Prompt
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

configure(openai_api_key=os.environ.get('KEY'), gemini_api_key=os.environ.get('G_KEY'))

generator = GeneratorFactory().create_generator(model_name='gpt-4o-mini')

prompt = Prompt().chat().system("you are a helpful assistent called Emillie")

print("Emillie: Hello, how can I be of assistence?\n")

async def get_res():
    user_input = input("You: ")
    print("\n")
    global prompt
    
    prompt = prompt.user(user_input)
    
    response =await generator.agenerate_response(prompt)
        
    print("Emillie:", response.get_text())

    prompt = prompt.assistant(response.get_text())
    
asyncio.run(get_res())