from Services import *
from functions import *
from openai import OpenAI
from google import genai

import config

def main():
    OpenAIClient = OpenAI(api_key= config.OPENAI_API_KEY)
    genAIClient =  genai.Client(api_key= config.GENAI_API_KEY)

    # response = genAIClient.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents="Explain how AI works",
    # )

    # print(response.text)

    # completion = OpenAIClient.chat.completions.create(
    #     model="gpt-4o-mini",
    #     store=True,
    #     messages=[
    #         {"role": "user", "content": "how many races are there?"}
    #     ]
    # )

    # print(completion.choices[0].message)

if __name__ == "__main__":
    main()
