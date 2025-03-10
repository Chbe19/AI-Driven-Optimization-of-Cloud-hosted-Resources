from Services import *
from Functions import *
from openai import OpenAI
import config

def main():
    client = OpenAI(api_key= config.OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
    )

    print(completion.choices[0].message)

if __name__ == "__main__":
    main()
