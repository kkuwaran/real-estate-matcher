import os 
import random

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings






# Get the directory where the current Python script is located
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


persist_directory = os.path.join(current_dir, "chromadb_data")
client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)


# Create a collection or get an existing one
collection = client.create_collection("real_estate_listings")





# Example function to generate embeddings (replace with actual embedding generation)
def generate_embedding(text):
    # Generate a random embedding with 300 dimensions (replace 300 with your desired size)
    embedding = [random.random() for _ in range(20)]  # Random float values between 0 and 1
    return embedding






# Example real estate listings
real_estate_listings = [
    {
        "id": 1,
        "title": "Cozy 2-Bedroom Condo in Downtown",
        "description": "This modern 2-bedroom, 1.5-bathroom condo offers an open floor plan, ...",
        "price": 320000,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "area_sqft": 850,
        "location": "San Francisco, CA",
        "property_type": "Condo",
        "year_built": 2010
    },
    {
        "id": 2,
        "title": "Spacious 4-Bedroom House with Garden",
        "description": "Beautiful house with large backyard, renovated kitchen, and 2-car garage. ...",
        "price": 715000,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "area_sqft": 2100,
        "location": "Austin, TX",
        "property_type": "House",
        "year_built": 2015
    },
    {
        "id": 3,
        "title": "Modern Studio Apartment Near University",
        "description": "Compact and efficient studio ideal for students, walkable to campus. This studio apartment ...",
        "price": 150000,
        "bedrooms": 0,
        "bathrooms": 1.0,
        "area_sqft": 450,
        "location": "Ann Arbor, MI",
        "property_type": "Apartment",
        "year_built": 2020
    }
]

# Insert each listing into the collection
for listing in real_estate_listings:
    # Generate embedding for the listing's description
    embedding = generate_embedding(listing["description"])

    # Insert the listing into the collection
    collection.add(
        ids=[str(listing["id"])],  # Use the listing's ID as a unique identifier
        embeddings=[embedding],
        metadatas=[listing],
        documents=[listing["description"]]
    )


# Query to retrieve all items from the collection
results = collection.query(
    query_embeddings=[],  # Empty list as you want to retrieve all items
    n_results=10  # Number of results to retrieve (set to a higher number if needed)
)

# Print the results to check the items in the collection
for result in results['documents']:
    print(result)

#==========================================================================================


# # Example real estate listing
# real_estate_listing = {
#     "id": 1,
#     "title": "Cozy 2-Bedroom Condo in Downtown",
#     "description": "This modern 2-bedroom, 1.5-bathroom condo offers an open floor plan...",
#     "price": 320000,
#     "bedrooms": 2,
#     "bathrooms": 1.5,
#     "area_sqft": 850,
#     "location": "San Francisco, CA",
#     "property_type": "Condo",
#     "year_built": 2010,
# }

# # Generate embedding for the listing's description
# embedding = generate_embedding(real_estate_listing["description"])

# # Insert the listing into the collection
# collection.add(
#     ids=[str(real_estate_listing["id"])],  # Use the listing's ID as a unique identifier
#     embeddings=[embedding],
#     metadatas=[real_estate_listing],
#     documents=[real_estate_listing["description"]]
# )