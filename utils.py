import os
import copy
import json
from pathlib import Path
from dotenv import load_dotenv
from IPython.display import display

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions


# Load variables from .env into environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")



class Database:

    DOCUMENT_TEMPLATE = (
        "{title}. {description} "
        "Neighborhood info: {neighborhood_description}. "
        "Location: {location}, Neighborhood: {neighborhood}. "
        "Type: {property_type}, Bedrooms: {bedrooms}, "
        "Bathrooms: {bathrooms}, Area: {area_sqft} sqft, "
        "Year built: {year_built}, Price: ${price}."
    )

    REAL_ESTATE_INFO_TEMPLATE = (
        "Property ID: {id}\n"
        "Title: {title}\n"
        "Description: {description}\n"
        "Neighborhood Info: {neighborhood_description}\n"
        "Location: {location}\n"
        "Neighborhood: {neighborhood}\n"
        "Property Type: {property_type}\n"
        "Year Built: {year_built} CE\n"
        "Price: ${price:,}\n"
        "Bedrooms: {bedrooms}\n"
        "Bathroom: {bathrooms}\n"
        "Area: {area_sqft} sq.ft.\n"
        "Listed Date: {listed_date}\n"
    )

    def __init__(self, persist_dir, collection_name, delete_flag=False, verbose=False):

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        self.verbose = verbose

        if delete_flag:
            self.delete_collection(collection_name)

        self.collection = self.get_collection(collection_name)
        self.count()

    
    def _get_collection_names(self):
        collection_names = [c.name for c in self.client.list_collections()]
        if self.verbose:
            print("collection_names:", collection_names)
        return collection_names


    def get_collection(self, collection_name: str):
        # Create a collection or get an existing one
        collection_names = self._get_collection_names()

        if collection_name in collection_names:
            print(f"*** Get '{collection_name}' collection ***")
            collection = self.client.get_collection(collection_name)
        else:
            print(f"*** Create '{collection_name}' collection ***")
            collection = self.client.create_collection(collection_name)
        
        return collection
    

    def delete_collection(self, collection_name: str):
        # Delete a collection
        print(f"*** Delete '{collection_name}' collection ***")
        self.client.delete_collection(collection_name)


    def count(self):
        # Check how many items are inside
        num_items = self.collection.count()
        print("Number of Documents:", num_items)
        return num_items
    

    def _setup_embedding_function(self, openai_model_name="text-embedding-3-small"):

        # Setup embedding function
        # https://platform.openai.com/docs/guides/embeddings/embedding-models
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_base=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY,
            model_name=openai_model_name,
        )

        self.collection.embedding_function = embedding_function


    def add_data(self, json_path: Path, embed_model=None):
    
        # Open and load the JSON file
        with json_path.open("r", encoding="utf-8") as f:
            real_estate_listings = json.load(f)
        print("No. of data:", len(real_estate_listings))

        if embed_model:
            self._setup_embedding_function(openai_model_name=embed_model)

        ids = [str(listing["id"]) for listing in real_estate_listings]

        documents = [
            self.DOCUMENT_TEMPLATE.format(**listing) 
            for listing in real_estate_listings
        ]

        metadatas = copy.deepcopy(real_estate_listings)

        # Add to the collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        self.count()

        # print(f"Inserted {len(documents)} listings into Chroma collection '{self.collection_name}'")


    def query_db(self, query_text: str, conditions: list, n_results: int = 1):

        query_outputs = self.collection.query(
            query_texts=[query_text],
            where={"$and": conditions},
            n_results=n_results,
        )

        if self.verbose:
            display(query_outputs)
        return query_outputs


    def _display_real_estate_info(self, real_estate_id):

        id_str = str(real_estate_id)
        item = self.collection.get(ids=[id_str])

        metadata = item['metadatas'][0]

        real_estate_info = self.REAL_ESTATE_INFO_TEMPLATE.format(**metadata)
        print(real_estate_info)


    def display_query_outputs(self, query_outputs, n_heads: int = 1):

        ids = query_outputs['ids']
        index = 0
        while index < min(len(ids), n_heads):
            id_str = ids[index]
            print("*" * 20, f"Rank-{index + 1}", "*" * 20)
            self._display_real_estate_info(id_str)
            print()
            index += 1