#!/usr/bin/env python
# coding: utf-8

# # Exercise: Generating one token at a time
# 
# In this exercise, we will get to understand how an LLM generates text--one token at a time, using the previous tokens to predict the following ones.
# 

# ## Step 1. Load a tokenizer and a model
# 
# First we load a tokenizer and a model from HuggingFace's transformers library. A tokenizer is a function that splits a string into a list of numbers that the model can understand.
# 
# In this exercise, all the code will be written for you. All you need to do is follow along!

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer

# To load a pretrained model and a tokenizer using HuggingFace, we only need two lines of code!
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")

# We create a partial sentence and tokenize it.
text   = "Udacity is the best place to learn about generative"
inputs = tokenizer(text, return_tensors="pt")

# Show the tokens as numbers, i.e. "input_ids"
print(inputs["input_ids"])


# In[2]:


tokenizer(text)
print(text)


# ## Step 2. Examine the tokenization
#
# Let's explore what these tokens mean!

# In[3]:


print(tokenizer.decode(257))


# In[4]:


# Show how the sentence is tokenized
import pandas as pd


def show_tokenization(inputs):
    return pd.DataFrame(
        [ (id, tokenizer.decode(id)) for id in inputs["input_ids"][0] ],
        columns=["id", "token"],
    )


print(show_tokenization(inputs))


# ### Subword tokenization
#
# The interesting thing is that tokens in this case are neither just letters nor just words. Sometimes shorter words are represented by a single token, but other times a single token represents a part of a word, or even a single letter. This is called subword tokenization.
#
# ## Step 2. Calculate the probability of the next token
#
# Now let's use PyTorch to calculate the probability of the next token given the previous ones.

# In[5]:


# Calculate the probabilities for the next token for all possible choices. We show the
# top 5 choices and the corresponding words or subwords for these tokens.

import torch

with torch.no_grad():
    logits        = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)


def show_next_token_choices(probabilities, top_n=5):
    return pd.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:top_n]


print(show_next_token_choices(probabilities))


# In[6]:


print(inputs)


# In[7]:


print(model(**inputs).logits.shape, model(**inputs).logits[:, -1, :].shape)


# In[8]:


print(torch.nn.functional.softmax(model(**inputs).logits[:, -1, :], dim=-1))


# In[9]:


print(probabilities)


# Interesting! The model thinks that the most likely next word is "programming", followed up closely by "learning".

# In[10]:


# Obtain the token id for the most probable next token
next_token_id = torch.argmax(probabilities).item()

print(f"Next token id: {next_token_id}")
print(f"Next token: {tokenizer.decode(next_token_id)}")


# In[11]:


# We append the most likely token to the text.
text = text + tokenizer.decode(next_token_id)
print(text)


# ## Step 3. Generate some more tokens
#
# The following cell will take `text`, show the most probable tokens to follow, and append the most likely token to text. Run the cell over and over to see it in action!

# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Press ctrl + enter to run this cell again and again to see how the text is generated.\n\nfrom IPython.display import Markdown, display\n\n# Show the text\nprint(text)\n\n# Convert to tokens\ninputs = tokenizer(text, return_tensors="pt")\n\n# Calculate the probabilities for the next token and show the top 5 choices\nwith torch.no_grad():\n    logits = model(**inputs).logits[:, -1, :]\n    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)\n\ndisplay(Markdown("**Next token probabilities:**"))\ndisplay(show_next_token_choices(probabilities))\n\n# Choose the most likely token id and add it to the text\nnext_token_id = torch.argmax(probabilities).item()\ntext = text + tokenizer.decode(next_token_id)\n')


# ## Step 4. Use the `generate` method

# In[13]:


get_ipython().run_cell_magic('time', '', '\nfrom IPython.display import Markdown, display\n\n# Start with some text and tokenize it\ntext = "Once upon a time, generative models"\ninputs = tokenizer(text, return_tensors="pt")\n\n# Use the `generate` method to generate lots of text\noutput = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)\n\n# Show the generated text\ndisplay(Markdown(tokenizer.decode(output[0])))\ntokenizer.decode(output[0])\n')


# In[ ]:





# ### That's interesting...
#
# You'll notice that GPT-2 is not nearly as sophisticated as later models like GPT-4, which you may have experience using. It often repeats itself and doesn't always make much sense. But it's still pretty impressive that it can generate text that looks like English.
#
# ## Congrats for completing the exercise! 🎉
#
# Give yourself a hand. And please take a break if you need to. We'll be here when you're refreshed and ready to learn more!
