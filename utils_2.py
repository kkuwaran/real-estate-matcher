import os
from typing import List, Optional, Tuple
from datetime import datetime
import json
from dotenv import load_dotenv
from pathlib import Path
from IPython.display import display
from pydantic import BaseModel

from openai import OpenAI

from utils_3 import show_section


# Load variables from .env into environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")



class BuyerPreferences(BaseModel):
    bedrooms: Optional[int]
    bathrooms: Optional[float]
    property_type: Optional[str]

    # Additional physical features
    area_min_sqft: Optional[int]   # minimum area (sq. ft)
    area_max_sqft: Optional[int]   # maximum area (sq. ft)
    building_max_age: Optional[int]  # max age in years
    building_min_year: Optional[int] # alternatively, built after year X
    
    # Lifestyle & location
    amenities: List[str] = []
    furnished: bool = False
    location: Optional[str]
    neighborhood_features: List[str] = []
    transportation: List[str] = []
    parking_required: bool = False
    pet_friendly_required: bool = False

    # Financial
    min_budget: Optional[int]
    max_budget: Optional[int]



class RealEstateConversations:

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

    def __init__(self, conv_json_path: Path, verbose=False):

        self.verbose = verbose
        self.client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

        self.conversations = self.load_conversations(conv_json_path)
        self.conv_ids = self.get_conv_ids()


    @staticmethod
    def load_conversations(conv_json_path: Path, id_key="conversation_id") -> dict:
        
        with conv_json_path.open("r", encoding="utf-8") as f:
            conversation_list = json.load(f)

        conversations = {conv[id_key]: conv for conv in conversation_list}
        return conversations
    

    def get_conv_ids(self):
        conv_ids = list(self.conversations.keys())
        print("Conversation IDs:", conv_ids, '\n')
        return conv_ids


    def _get_conversation_str(self, conv_id: int):

        assert conv_id in self.conv_ids, "invalid conversation id"
        conv = self.conversations[conv_id]

        conversation_text = ""
        for msg in conv["messages"]:
            role = msg['role'].capitalize()
            conversation_text += f"{role}: {msg['text']}\n"

        if self.verbose:
            show_section(f"Conversation {conv_id}", conversation_text)

        return conversation_text
    

    def _extract_preferences(self, conversation_text: str):

        system_prompt = "You are a real estate preference parser."
        user_prompt = f"Extract buyer preferences from the conversation below and fill the JSON fields:\n{conversation_text}"

        response = self.client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=BuyerPreferences
        )

        content = response.choices[0].message.content
        content_dict = json.loads(content)

        if self.verbose:
            show_section("Buyer Preferences", content_dict, use_display=True)

        return BuyerPreferences(**content_dict)
    

    def yes_no(self, value: bool) -> str:
        return "Yes" if value else "No"


    def join_or_any(self, values: list[str]) -> str:
        return ", ".join(values) if values else "Any"
    

    def _get_query_text(self, prefs: BuyerPreferences) -> str:
        
        buyer_preferences = prefs.model_dump()
        values = {
            "property_type": buyer_preferences["property_type"],
            "amenities": self.join_or_any(buyer_preferences["amenities"]),
            "furnished": self.yes_no(buyer_preferences["furnished"]),
            "location": buyer_preferences["location"] or "Any",
            "neighborhood_features": self.join_or_any(buyer_preferences["neighborhood_features"]),
            "transportation": self.join_or_any(buyer_preferences["transportation"]),
            "parking_required": self.yes_no(buyer_preferences["parking_required"]),
            "pet_friendly_required": self.yes_no(buyer_preferences["pet_friendly_required"]),
        }

        query_text = self.QUERY_TEXT_TEMPLATE.format(**values)

        if self.verbose:
            show_section("Query Text", query_text)

        return query_text
    

    def _get_conditions(self, prefs: BuyerPreferences):

        # Get the current date and time, and then extract the year attribute
        current_year = datetime.now().year


        # n_bedroom = content_dict['bedrooms']
        # n_bathroom = content_dict['bathrooms']
        # min_budget, max_budget = content_dict['min_budget'], content_dict['max_budget']
        # min_area, max_area = content_dict['area_min_sqft'], content_dict['area_max_sqft']
        # building_max_age = content_dict['building_max_age']
        # building_min_year = content_dict['building_min_year']

        conditions = []
        if prefs.bedrooms:
            conditions += [{"bedrooms": prefs.bedrooms}]
        if prefs.bathrooms:
            conditions += [{"bathrooms": {"$gte": prefs.bathrooms - 1}}, {"bathrooms": {"$lte": prefs.bathrooms + 1}}]
        if prefs.min_budget:
            conditions += [{"price": {"$gte": prefs.min_budget}}]
        if prefs.max_budget:
            conditions += [{"price": {"$lte": prefs.max_budget}}]
        if prefs.area_min_sqft:
            conditions += [{"area_sqft": {"$gte": prefs.area_min_sqft}}]
        if prefs.area_max_sqft:
            conditions += [{"area_sqft": {"$lte": prefs.area_max_sqft}}]
        if prefs.building_min_year:
            conditions += [{"year_built": {"$gte": prefs.building_min_year}}]
        if prefs.building_max_age:
            conditions += [{"year_built": {"$gte": current_year - prefs.building_max_age}}]

        if self.verbose:
            show_section("Filter Conditions", conditions, use_display=True)

        return conditions
    

    def get_query_and_conditions(self, conv_id: int):

        conversation_text = self._get_conversation_str(conv_id)
        buyer_prefs = self._extract_preferences(conversation_text)
        query_text = self._get_query_text(buyer_prefs)
        conditions = self._get_conditions(buyer_prefs)
        return query_text, conditions
    

