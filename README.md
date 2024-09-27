# ChatFusion ⚡

A flexible and powerful Python library for interacting with various AI language models, including OpenAI's GPT and Google's Gemini.

## Features

- Support for multiple AI providers (OpenAI and Google Gemini and more coming) as well as capability of adding your own
- Easy-to-use API for generating responses
- Flexible prompt system supporting both single-message and chat-based interactions immutable data structure
- Handling of text and file inputs
- Streaming support for real-time responses
- Customizable generation parameters

## Installation

```bash
pip install chatbots
```
and install the AI libraries you plan on using, for now google-generativeai and openai have built in support

## Usage

```python
from chatbots import configure, GeneratorFactory, Prompt

configure(gemini_api_key="YOUR_API_KEY", openai_api_key= 'YOUR_API_KEY')

factory = GeneratorFactory()

gemini_model = factory.create_generator(model_name='gemini-1.5-pro-latest')
gpt_4o = factory.create_generator(model_name='gpt-4o-mini')

prompt = Prompt().text('Hi, how are you?')

response = gemini_model.generate_response(prompt)

print(response.text())

response = gpt_4o.generate_response(prompt)

print(response.text())

prompt = Prompt().chat().system('Your name is mason')
prompt = prompt.user('whats your name?')

response = gemini_model.generate_response(prompt)

print(response.text())
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.