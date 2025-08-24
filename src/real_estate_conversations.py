import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI

from real_estate_database import Database
from utility import show_section


# Load variables from .env into environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")



class BuyerPreferences(BaseModel):
    """A Pydantic model representing buyer preferences for real estate properties."""

    bedrooms: Optional[int] = None   # number of bedrooms
    bathrooms: Optional[float] = None  # number of bathrooms (can be fractional, e.g., 1.5)
    property_type: Optional[str] = None  # e.g., "apartment", "condo", "house"

    # Additional physical features
    area_min_sqft: Optional[int] = None   # minimum area (sq. ft)
    area_max_sqft: Optional[int] = None   # maximum area (sq. ft)
    building_max_age: Optional[int] = None  # maximum building age in years
    building_min_year: Optional[int] = None  # alternatively, built after year X
    
    # Lifestyle & location
    amenities: List[str] = Field(default_factory=list)  # e.g., ["pool", "gym", "garden"]
    furnished: bool = False                             # whether furnished is required
    location: Optional[str] = None                      # general location string (e.g., city or district)
    neighborhood_features: List[str] = Field(default_factory=list)  # e.g., ["quiet", "family-friendly"]
    transportation: List[str] = Field(default_factory=list)         # e.g., ["near subway", "bus line access"]
    parking_required: bool = False                      # parking requirement
    pet_friendly_required: bool = False                 # must allow pets

    # Financial
    min_budget: Optional[int] = None   # minimum budget in USD (or local currency)
    max_budget: Optional[int] = None   # maximum budget in USD (or local currency)



class RealEstateConversations:
    """
    Manages real estate buyer-seller conversations and extracts buyer preferences
    using an LLM (OpenAI chat model).

    This class:
      - Loads conversation data from JSON
      - Provides access to conversation IDs and text
      - Extracts buyer preferences from conversations into a structured model
    """

    QUERY_TEXT_TEMPLATE = (
        "- Property type: {property_type}\n"
        "- Amenities: {amenities}\n"
        "- Furnished: {furnished}\n"
        "- Location: {location}\n"
        "- Neighborhood features: {neighborhood_features}\n"
        "- Transportation preferences: {transportation}\n"
        "- Parking required: {parking_required}\n"
        "- Pet friendly required: {pet_friendly_required}"
    )

    # Default condition relaxation settings
    RELAXATION_CONFIG = {
        "bathrooms_tolerance": 0.5,     # initial tolerance for bathrooms
        "budget_pct": 0.15,             # per relaxation level
        "area_sqft_delta": 200,         # sq.ft change per level
        "max_building_relax_level": 2,  # max level to enforce year/age constraints
    }


    def __init__(self, conv_json_path: Path, model="gpt-4o-2024-08-06", verbose: bool = False) -> None:
        """Initialize RealEstateConversations."""

        self.client: OpenAI = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
        self.model = model
        self.verbose: bool = verbose

        self.conversations: Dict[int, Dict[str, Any]] = self._load_conversations(conv_json_path)
        self.conv_ids: List[int] = self._get_conv_ids()

    
    # ===============================
    # Private helpers
    # ===============================

    @staticmethod
    def _yes_no(value: bool) -> str:
        """Convert a boolean value into a human-readable 'Yes' or 'No' string."""
        return "Yes" if value else "No"


    @staticmethod
    def _join_or_any(values: List[str]) -> str:
        """Join a list of strings into a comma-separated string, or return 'Any' if the list is empty."""
        return ", ".join(values) if values else "Any"


    @staticmethod
    def _load_conversations(conv_json_path: Path, id_key: str = "conversation_id") -> Dict[int, Dict[str, Any]]:
        """
        Load conversations from a JSON file and return as a dictionary.
            - conv_json_path (Path): Path to JSON file.
            - id_key (str): Key used for conversation IDs inside JSON objects.
            - conversations (dict): Mapping from conversation_id to conversation content.
        """

        with conv_json_path.open("r", encoding="utf-8") as f:
            conversation_list: List[Dict[str, Any]] = json.load(f)

        conversations = {conv[id_key]: conv for conv in conversation_list}
        return conversations
    

    def _get_conv_ids(self) -> List[int]:
        """Retrieve all conversation IDs."""

        conv_ids = list(self.conversations.keys())
        print("Conversation IDs:", conv_ids, '\n')
        return conv_ids

    
    # ===============================
    # Public methods
    # ===============================

    def get_conversation_text(self, conv_id: int) -> str:
        """Retrieve the conversation text for a given conversation ID."""

        if conv_id not in self.conv_ids:
            raise ValueError(f"Invalid conversation ID: {conv_id}")

        conv = self.conversations[conv_id]

        # Format messages into readable text
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['text']}"
            for msg in conv["messages"]
        )

        if self.verbose:
            show_section(f"Conversation {conv_id}", conversation_text)

        return conversation_text
    

    def extract_preferences(self, conv_id: int) -> Optional[BuyerPreferences]:
        """Extract buyer preferences from a conversation using an LLM."""

        conversation_text = self.get_conversation_text(conv_id)

        system_prompt = "You are a real estate preference parser."
        user_prompt = (
            "Extract buyer preferences from the conversation below and "
            "fill the JSON fields:\n"
            f"{conversation_text}"
        )

        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=BuyerPreferences,
        )

        content = response.choices[0].message.content

        try:
            content_dict = json.loads(content)
            prefs = BuyerPreferences(**content_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            if self.verbose:
                show_section("Preference Extraction Error", str(e))
            return None

        if self.verbose:
            show_section("Buyer Preferences", prefs.model_dump(), use_display=True)

        return prefs


    def build_query_text(self, prefs: BuyerPreferences) -> str:
        """Build a textual query from buyer preferences to be used in a semantic search."""

        buyer_dict = prefs.model_dump()

        values = {
            "property_type": buyer_dict["property_type"] or "Any",
            "amenities": self._join_or_any(buyer_dict["amenities"]),
            "furnished": self._yes_no(buyer_dict["furnished"]),
            "location": buyer_dict["location"] or "Any",
            "neighborhood_features": self._join_or_any(buyer_dict["neighborhood_features"]),
            "transportation": self._join_or_any(buyer_dict["transportation"]),
            "parking_required": self._yes_no(buyer_dict["parking_required"]),
            "pet_friendly_required": self._yes_no(buyer_dict["pet_friendly_required"]),
        }

        query_text = self.QUERY_TEXT_TEMPLATE.format(**values)

        if self.verbose:
            show_section("Query Text", query_text)

        return query_text
    

    def build_filter_conditions(self, prefs: BuyerPreferences, relaxation_level: int = 0) -> List[dict]:
        """Convert buyer preferences into a list of filter conditions for a structured query."""

        # Get the current date and time, and then extract the year attribute
        current_year = datetime.now().year

        cfg = self.RELAXATION_CONFIG
        conditions = []

        if prefs.bedrooms is not None:
            conditions.append({"bedrooms": prefs.bedrooms})

        if prefs.bathrooms is not None:
            tol = cfg["bathrooms_tolerance"] + relaxation_level
            conditions.append({"bathrooms": {"$gte": prefs.bathrooms - tol}})
            conditions.append({"bathrooms": {"$lte": prefs.bathrooms + tol}})

        if prefs.min_budget is not None:
            min_budget = int(prefs.min_budget * (1 - cfg["budget_pct"] * relaxation_level))
            conditions.append({"price": {"$gte": min_budget}})
        if prefs.max_budget is not None:
            max_budget = int(prefs.max_budget * (1 + cfg["budget_pct"] * relaxation_level))
            conditions.append({"price": {"$lte": max_budget}})

        if prefs.area_min_sqft is not None:
            conditions.append({"area_sqft": {"$gte": max(0, prefs.area_min_sqft - cfg["area_sqft_delta"] * relaxation_level)}})
        if prefs.area_max_sqft is not None:
            conditions.append({"area_sqft": {"$lte": prefs.area_max_sqft + cfg["area_sqft_delta"] * relaxation_level}})

        if prefs.building_min_year is not None and relaxation_level < cfg["max_building_relax_level"]:
            conditions.append({"year_built": {"$gte": prefs.building_min_year}})
        if prefs.building_max_age is not None and relaxation_level < cfg["max_building_relax_level"]:
            conditions.append({"year_built": {"$gte": current_year - prefs.building_max_age}})

        return conditions

        # # Get the current date and time, and then extract the year attribute
        # current_year = datetime.now().year

        # conditions = []
        # if prefs.bedrooms is not None:
        #     conditions.append({"bedrooms": prefs.bedrooms})
        # if prefs.bathrooms is not None:
        #     # Allow +/-1 range for bathrooms
        #     conditions.append({"bathrooms": {"$gte": prefs.bathrooms - 1}})
        #     conditions.append({"bathrooms": {"$lte": prefs.bathrooms + 1}})
        # if prefs.min_budget is not None:
        #     conditions.append({"price": {"$gte": prefs.min_budget}})
        # if prefs.max_budget is not None:
        #     conditions.append({"price": {"$lte": prefs.max_budget}})
        # if prefs.area_min_sqft is not None:
        #     conditions.append({"area_sqft": {"$gte": prefs.area_min_sqft}})
        # if prefs.area_max_sqft is not None:
        #     conditions.append({"area_sqft": {"$lte": prefs.area_max_sqft}})
        # if prefs.building_min_year is not None:
        #     conditions.append({"year_built": {"$gte": prefs.building_min_year}})
        # if prefs.building_max_age is not None:
        #     # Convert max building age to minimum construction year
        #     conditions.append({"year_built": {"$gte": current_year - prefs.building_max_age}})

        # if self.verbose:
        #     show_section("Filter Conditions", conditions, use_display=True)

        # return conditions
    

    # def get_query_text_and_conditions(self, conv_id: int) -> Tuple[str, List[dict]]:
    #     """
    #     Given a conversation ID, extract buyer preferences and generate both:
    #     - Query text for semantic search
    #     - Filter conditions for structured query
    #     """

    #     buyer_prefs = self.extract_preferences(conv_id)

    #     query_text = self.build_query_text(buyer_prefs)
    #     conditions = self.build_filter_conditions(buyer_prefs)

    #     return query_text, conditions
    



    def query_with_progressive_relaxation(
        self,
        conv_id: int,
        database: Database,
        n_results: int = 3,
        max_relaxation_level: int = 5,
    ) -> List[str]:
        """
        Perform a semantic + structured database query based on buyer preferences,
        progressively relaxing filter conditions until enough results are found.

        Args:
            conv_id (int): ID of the conversation from which to extract buyer preferences.
            database (Database): Instance of the Database class to query.
            n_results (int, optional): Minimum number of results to return. Defaults to 3.
            max_relaxation_level (int, optional): Maximum number of relaxation iterations. Defaults to 5.

        Returns:
            List[str]: List of real estate IDs matching the query.
                    Returns an empty list if preferences cannot be extracted or no matches are found.
        """

        # Extract buyer preferences from conversation
        buyer_prefs = self.extract_preferences(conv_id)
        if buyer_prefs is None:
            show_section("Error", "Failed to extract buyer preferences from conversation.")
            return []

        # Build the query text for semantic search
        query_text = self.build_query_text(buyer_prefs)

        ids = []

        # Progressive relaxation loop
        for level in range(max_relaxation_level + 1):
            print("=" * 35, f"\nRelaxation Level {level}: Querying ...")

            # Build structured filters and perform database query
            conditions = self.build_filter_conditions(buyer_prefs, relaxation_level=level)
            query_outputs = database.query(query_text, conditions, n_results=n_results)

            # Check if at least n_results were found
            ids = database.extract_ids_from_query_outputs(query_outputs)
            if len(ids) >= n_results:
                break  # Stop relaxing once we have enough results

        # Display the query results
        database.display_results_from_ids(ids, n_heads=n_results)

        return ids
