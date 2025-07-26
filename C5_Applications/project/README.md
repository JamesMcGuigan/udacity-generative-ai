# HomeMatch

## Step 1: Setting Up the Python Application
```
conda activiate gymnasium
pip install -r requirements.txt
pip uninstall pydantic
pip install "pydantic<2.0"  # BUGFIX
```

## Step 2: Generating Real Estate Listings
listings.py calls ChatGPT 10 times in a loop then writes YAML to filesystem
```
python3 ./listings.py
cat     ./listings.txt
```

## Step 3: Storing Listings in a Vector Database
ChromaDB.py reads listings.txt YAML and imports into ChromaDB
BUG: ChromaDB does not seem to persist to filesystem, thus must be reimported each runtime 
```
python3 ./ChromaDB.py
```

## Step 4: Building the User Preference Interface
## Step 5: Searching Based on Preferences
## Step 6: Personalizing Listing Descriptions
Hardcoded user input questions are converted into a ChromaDB embedding search.
Top 3 results are then augmented with ChatGPT estate agent review  
```
python3 ./HomeMatch.py
```

