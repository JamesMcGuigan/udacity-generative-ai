import os
import yaml
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

model_name = "gpt-3.5-turbo-instruct"  # -instruct required for OpenAI()
temperature = 1.0
llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens=3456)

# BUGFIX: ChatGPT lacks tokens to print all 10 out in the single response
# BUGFIX: ChatGPT fails to output any content if we ask for 1 response at a time
# BUGFIX: ChatGPT in a 10 loop works correctly if we ask for 2 at a time but expect 1 output each time
prompt = """
Generate 2 diverse and realistic real estate listing in the following format:

---
Neighborhood: [Neighborhood Name]
Price: [Price]
Bedrooms: [Number of Bedrooms]
Bathrooms: [Number of Bathrooms]
House Size: [House Size]

Description: [Description of the property]

Neighborhood Description: [Description of the neighborhood]
---

Example:
---
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
---
"""

# Convert --- separated yaml into objects
def read_listings(file='listings.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        listings = f.read().strip().split('---')
        parsed_listings = [ yaml.safe_load(listing)
                            for listing in listings
                            if listing.strip() ]
        # print(parsed_listings)
        return parsed_listings


if __name__ == '__main__':
    # Generate the listings
    responses = []
    for i in range(10):
        response = llm(prompt)
        responses.append( response )

    with open('listings.txt', 'w', encoding='utf-8') as f:
        f.write("\n---\n".join(responses))
        print("\n---\n".join(responses))
        print('wrote: listings.txt')


