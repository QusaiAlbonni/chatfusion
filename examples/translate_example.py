from chatfusion import configure
from chatfusion import GeneratorFactory
from chatfusion import Prompt
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

configure(openai_api_key=os.environ.get('KEY'), gemini_api_key=os.environ.get('G_KEY'))

generator = GeneratorFactory().create_generator(model_name='o1-mini')

prompt = Prompt().translate("die katze ist sehr schon", extra="Note: only output the translated text do not add anything else")

response =generator.generate_response(prompt)
        
print("Translator:", response.get_text())