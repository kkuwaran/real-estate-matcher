import os
import copy
import json
from typing import Any, List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.api.types import Documents
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions

from utility import show_section


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
    

    # ===============================
    # Private helpers
    # ===============================
    
    def _get_collection_names(self) -> List[str]:
        """Return all available collection names."""

        collection_names = [c.name for c in self.client.list_collections()]
        if self.verbose:
            print("Available collections:", collection_names)
        return collection_names


    def _setup_embedding_function(self, openai_model_name: str = "text-embedding-3-small") -> None:
        """Attach an OpenAI embedding function to the current collection."""

        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_base=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY,
            model_name=openai_model_name,
        )
        self.collection.embedding_function = embedding_function
        if self.verbose:
            print(f"Embedding function set up with model: {openai_model_name}")

    
    # ===============================
    # Public methods
    # ===============================

    def get_collection(self, collection_name: str):
        """Retrieve or create a collection by name."""

        if collection_name in self._get_collection_names():
            print(f"*** Retrieving existing collection: '{collection_name}' ***")
            return self.client.get_collection(collection_name)
        else:
            print(f"*** Creating new collection: '{collection_name}' ***")
            return self.client.create_collection(collection_name)
        
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by name."""

        print(f"*** Deleting collection: '{collection_name}' ***")
        self.client.delete_collection(collection_name)


    def count(self) -> int:
        """Return the number of items in the current collection."""

        num_items = self.collection.count()
        print("Number of documents in collection:", num_items)
        return num_items


    def add_data(self, json_path: Path, embed_model: Optional[str] = None) -> None:
        """Load JSON data and add it to the collection."""

        # Load listings
        with json_path.open("r", encoding="utf-8") as f:
            real_estate_listings = json.load(f)
        print(f"Loaded {len(real_estate_listings)} records")

        # Setup embedding if requested
        if embed_model:
            self._setup_embedding_function(openai_model_name=embed_model)

        # Prepare ChromaDB fields
        ids = [str(listing["id"]) for listing in real_estate_listings]
        documents: Documents = [
            self.DOCUMENT_TEMPLATE.format(**listing) 
            for listing in real_estate_listings
        ]
        metadatas = copy.deepcopy(real_estate_listings)

        # Insert into ChromaDB
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

        self.count()


    def query(self, query_text: str, conditions: List[Dict[str, Any]], n_results: int = 1) -> Dict[str, Any]:
        """
        Query the database collection with filtering conditions.
            - query_text (str): The natural language query string.
            - conditions (List[Dict[str, Any]]): Filtering conditions for the query.
            - n_results (int, optional): Number of results to return. Defaults to 1.
        """

        query_outputs = self.collection.query(
            query_texts=[query_text],
            where={"$and": conditions},
            n_results=n_results,
        )

        if self.verbose:
            show_section("Query Outputs", query_outputs, use_display=True)

        return query_outputs


    def fetch_real_estate_info(self, real_estate_id: str) -> str:
        """Retrieve formatted metadata for a specific real estate entry."""
        
        id_str = str(real_estate_id)
        item = self.collection.get(ids=[id_str])

        # Extract metadata for the first matching entry
        metadata = item["metadatas"][0]

        # Use predefined template to render information
        real_estate_info = self.REAL_ESTATE_INFO_TEMPLATE.format(**metadata)
        return real_estate_info


    def display_results(self, query_outputs: Dict[str, Any], n_heads: int = 1) -> None:
        """
        Display top-ranked query results with formatted real estate information.
            - query_outputs (Dict[str, Any]): Results from the `query` method.
            - n_heads (int, optional): Number of top-ranked results to display. Defaults to 1.
        """

        ids = query_outputs.get("ids", [[]])[0]  # Safely extract list of IDs
        for rank, real_estate_id in enumerate(ids[:n_heads], start=1):
            real_estate_info = self.fetch_real_estate_info(real_estate_id)
            show_section(f"Rank-{rank}", real_estate_info)
