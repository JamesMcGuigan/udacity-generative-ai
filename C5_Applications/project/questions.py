# Step 4: Building the User Preference Interface
#
# Collect buyer preferences, such as the number of bedrooms, bathrooms, location, and other specific requirements
# from a set of questions or telling the buyer to enter their preferences in natural language.
# You can hard-code the buyer preferences in questions and answers, or collect them interactively however you'd like, example:

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from ChromaDB import get_embedding, collection  # import regenerates collection
import json

questions = [
    "How big do you want your house to be?" 
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

def search_chromadb(user_input):
    # Generate an embedding for the user's input
    user_embedding = get_embedding(user_input)

    # Search the ChromaDB collection using the embedding
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=3  # Number of results to return
    )
    return results


# Step 6: Personalizing Listing Descriptions
# LLM Augmentation: For each retrieved listing, use the LLM to augment the description, tailoring it to resonate with the buyer’s specific preferences. This involves subtly emphasizing aspects of the property that align with what the buyer is looking for.
# Maintaining Factual Integrity: Ensure that the augmentation process enhances the appeal of the listing without altering factual information.
def personalize_listing_description(user_preferences: str, listing: dict) -> str:
    """
    Rewrites the property listing description to highlight features relevant to buyer preferences.
    """
    llm = ChatOpenAI(model_name="gpt-4o")
    messages = [
        SystemMessage(content="You are a property listing assistant. Only augment factual details..."),
        HumanMessage(content=f"Property Description:\n{listing}"),
        HumanMessage(content=f"Buyer Preferences:\n{user_preferences}")
    ]
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else response


if __name__ == "__main__":
    # Step 5: Searching Based on Preferences
    #
    # Buyer Preference Parsing: Implement logic to interpret and structure these preferences for querying the vector database.
    # Semantic Search Implementation: Use the structured buyer preferences to perform a semantic search on the vector database, retrieving listings that most closely match the user's requirements.
    # Listing Retrieval Logic: Fine-tune the retrieval algorithm to ensure that the most relevant listings are selected based on the semantic closeness to the buyer’s preferences.

    user_input = "\n".join([ f"{question} {answer}" for question, answer in zip(questions, answers) ])
    search_results = search_chromadb(user_input)
    for result in search_results['metadatas'][0]:
        review = personalize_listing_description(user_input, result)
        result['Agent Review'] = review
        print(json.dumps(result, indent=4))





