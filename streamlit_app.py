import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid
import random
import pandas as pd
from docx import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langsmith import trace, Client
import functools

# Load environment variables
load_dotenv()

# Load API keys and set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Bents-Woodworking-Assistant"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
langsmith_client = Client(api_key=LANGCHAIN_API_KEY)

# Pinecone index names
TRANSCRIPT_INDEX_NAME = "bents-woodworking"
PRODUCT_INDEX_NAME = "bents-woodworking-products"

# Initialize Pinecone indexes
for INDEX_NAME in [TRANSCRIPT_INDEX_NAME, PRODUCT_INDEX_NAME]:
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # OpenAI embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

transcript_index = pc.Index(TRANSCRIPT_INDEX_NAME)
product_index = pc.Index(PRODUCT_INDEX_NAME)

# YouTube video links
YOUTUBE_LINKS = {
    "Basics of Cabinet Building": "https://www.youtube.com/watch?v=Oeu7ogH2NZU&t=3910s",
    "Graco Ultimate Sprayer": "https://www.youtube.com/watch?v=T8BIpNzdh7M&t=264s",
    "Festool LR32 system": "https://www.youtube.com/watch?v=EO62T1LHdNA",
    "Harvey Gyro Air G700 - NORDFAB Duct Work": "https://www.youtube.com/watch?v=9MNXE_VFsIU",
"Pocket hole jigs are in trouble": "https://www.youtube.com/watch?v=ewap7x7WX2M",
"Assembly Table and Miter Saw Station - New Shop Part 7": "https://www.youtube.com/watch?v=pIJAnRuFKts",
"Live - Episode 6": "https://www.youtube.com/watch?v=QGkFQf1YVW8",
"Live - Episode 10 - 15 May 2021": "https://www.youtube.com/watch?v=zs_eBZuBlHA",
"Unleashing the Power of Narrow Crown Staples for Ultimate Panel Holding Strength": "https://www.youtube.com/shorts/ZEHu-pJSuTg",
"Unlock the Secret: Adjustable Clips for Ultimate Perfection!": "https://www.youtube.com/shorts/2OLLpDx8Yoo",
"These drawers will change your life": "https://www.youtube.com/watch?v=XE3E5l__qKA",
"Bents Woodworking Live - Episode 5": "https://www.youtube.com/watch?v=rupruQkLb3g",
"The most used tool in your shop": "https://www.youtube.com/watch?v=IazK9aL2yV4",
"Was LIFEPROOF Flooring a waste of Money?": "https://www.youtube.com/watch?v=_SsJbHInhC0",
"Bents Woodworking Live - Episode 1": "https://www.youtube.com/watch?v=6WsUldHQmNU",
"Things I like and do not like part 1": "https://www.youtube.com/watch?v=9MYd0uhcxoc",
"You might have the wrong one": "https://www.youtube.com/watch?v=cZ9-HoUwMI8",
"This is so confusing for most people": "https://www.youtube.com/watch?v=2L2ch7Rxns0",
"Live - Episode 8 - 30 Jan 2021": "https://www.youtube.com/watch?v=I0Y7lDPwy1E",
"Watch this before buying a festool track saw": "https://www.youtube.com/watch?v=sVWg53M_fmg",
"Unveiling the Secrets of Effortlessly Building Stunning Cabinetry": "https://www.youtube.com/shorts/dXVR84SrUvs",
"The Ultimate Tool Upgrade: How a Sliding Table Saw Changed Everything!": "https://www.youtube.com/watch?v=HkmTMAq1S5c",
"Bents Woodworking LIVE - 16 Oct 2021": "https://www.youtube.com/watch?v=G4hAMlwD5Jo",
"Bents Woodworking and More Live - Episode 4": "https://www.youtube.com/watch?v=2ejCdy6cbbk",
"The secret to a black finish": "https://www.youtube.com/watch?v=WwtbGfUZEWk",
"Track Saw Square Comparison //TSO Products//Bench Dogs UK//Woodpeckers Tools//Insta Rail Square": "https://www.youtube.com/watch?v=fBbrXqjXMrs",
"Discover the Game-Changing Power of Undermount Drawer Slides!": "https://www.youtube.com/shorts/nU9H9mX_-4Q",
"Easy Cabinet Installation Hacks for a Face Frame: Create Space and Achieve Overlay": "https://www.youtube.com/shorts/YvAV8oxNhGA",
"Bents Woodworking Live - Episode 2.5": "https://www.youtube.com/watch?v=J7pDDUfhr6c",
"Live - Episode 9 - 13 Feb 2021": "https://www.youtube.com/watch?v=v4lfULOuSxw",
"Genius Trick for Easy Drawer Slide Installation!": "https://www.youtube.com/shorts/JsT9iz9qYRs",
"Mastering Basic Drawer Box Construction: Avoiding Costly Measurement Mistakes": "https://www.youtube.com/shorts/RttpxCSQksU",
"Perfectly Fit Your Boards with Easy Drawer Slide Installation!": "https://www.youtube.com/shorts/EY42aY_Aw7I",
"Ultimate shop organization || PLANS AVAILABLE": "https://www.youtube.com/watch?v=1Mt_a2Lhcr8",
"Using A Table Saw To CUT RABBETS // WOODWORKING TIPS": "https://www.youtube.com/watch?v=_98dKhvfy98",
"What every domino owner needs to know": "https://www.youtube.com/watch?v=ZV2QsAmx76w",
"Walnut Dining Table Build // Plans Available": "https://www.youtube.com/watch?v=-yRhxAart6g",
"Using SketchUp To Design Woodworking Shop - New Shop Part 1": "https://www.youtube.com/watch?v=LcKjCFmmoTE",
"What is The Difference Between Festools Dust Extractors?": "https://www.youtube.com/watch?v=vNq69CotyvU",
"Woodpeckers Tools - Most and Least Used": "https://www.youtube.com/watch?v=xNUo9_eZEcs",
"Bents Woodworking Live - Episode 3": "https://www.youtube.com/watch?v=56I1zyhiuW4",
"Streamline Your DIY Projects with This Game-Changing Tip!": "https://www.youtube.com/shorts/btPy9eYTWJM",
"The Ultimate Guide to Perfect Measurements - Unlock Your DIY Potential!": "https://www.youtube.com/shorts/V6M7oAdVtMw",
"Bents Woodworking Live - Episode 2": "https://www.youtube.com/watch?v=c0WI6Rhzngc",
"You are overthinking it": "https://www.youtube.com/watch?v=pjOHdjCggPY",
"Track saw with 2 blades,  cuts better than your table saw": "https://www.youtube.com/watch?v=n-bjP2hGsSE",
"Something everyone should see": "https://www.youtube.com/watch?v=GA_9G4CY7Xw",
"Metric vs Imperial For Woodworking": "https://www.youtube.com/watch?v=1cVFmQbORBA",
"The biggest advancement in dust collection": "https://www.youtube.com/watch?v=hGO_sEyYqrs",
"Overview of the Festool TS 55 Track Saw": "https://www.youtube.com/watch?v=C9e6iEYWEVk",
"My 3 Car Garage Woodworking Shop In Detail - Woodworking Shop Tour 2021 -  Woodworking Wisdom": "https://www.youtube.com/watch?v=hQB7yqFSiIk",
"Stop trying to use every inch": "https://www.youtube.com/watch?v=nGrxaXvMWcs",
"Router Bit Storage Cabinet // Plans Available": "https://www.youtube.com/watch?v=J7XQy1hvxUc",
"The BEST under mount drawer slides": "https://www.youtube.com/watch?v=Af7NfKmkLuE",
"Stop wasting your money on the wrong ones": "https://www.youtube.com/watch?v=vJ6RG9sgKEE",
"Start Woodworking": "https://www.youtube.com/watch?v=sw1q8yPDop0",
"The fence unlike any other": "https://www.youtube.com/watch?v=XeaTeksLvbw",
"STOP struggling to get level cabinets": "https://www.youtube.com/watch?v=4Xh3dmfrrd8",
"My Process of Milling Lumber": "https://www.youtube.com/watch?v=LQ1bLCt4C90",
"The 5 TSO tools you cannot live without": "https://www.youtube.com/watch?v=WzP4z_hiDHQ",
"The BEST Table Saw Fence on the Market - Prove Me Wrong": "https://www.youtube.com/watch?v=JQTilBkTmEs",
"Pro cabinet makers do not want you to know how to do this": "https://www.youtube.com/watch?v=yptSgnx7V1I",
"Scribing A Cabinet To A Wall [Bents Woodworking]": "https://www.youtube.com/watch?v=hzi5V05eMlM",
"Selling my router table FOR THIS!": "https://www.youtube.com/watch?v=rVIesyJChP8",
"Other brands are in trouble": "https://www.youtube.com/watch?v=sXNGjsew_Cc",
"The Basics of Making Cabinets": "https://www.youtube.com/watch?v=mNQi2UOFmSo",
"The Best Table Saw Accessory": "https://www.youtube.com/watch?v=6m6bSLgfpb0",
"The good and the bad part 2": "https://www.youtube.com/watch?v=k-Jc233SM6Q",
"Leather Woodworking Apron - Dragonfly Woodworking and Leather": "https://www.youtube.com/watch?v=UEBxwUJex1g",
"Shop Storage Cabinet // Plans Available": "https://www.youtube.com/watch?v=hzpisSiPVjI",
"My shop is soundproof": "https://www.youtube.com/watch?v=YIkwN__pJ1w",
"STOP overbuilding cabinets": "https://www.youtube.com/watch?v=KmM6DOy0aMc",
"My favorite track saw so far": "https://www.youtube.com/watch?v=x_dHqsFmKwU",
"One feature that changes everything": "https://www.youtube.com/watch?v=JYbVWhsPdhU",
"People Told Me My Garage Door Would Explode": "https://www.youtube.com/watch?v=2zVejCdY9BU",
"Splines Using a Biscuit Joiner  I  Bents Woodworking": "https://www.youtube.com/watch?v=dMdJQZQO2zQ",
"Product Review: Festool ETS 125 and Rotex 125": "https://www.youtube.com/watch?v=QltohcYi9uM",
"So many people buy the wrong one": "https://www.youtube.com/watch?v=k2Glu6encWI",
"Leather By Dragonfly Apron Review": "https://www.youtube.com/watch?v=DEEILiKw3lo",
"Moving A Woodworking Shop - New Shop Part 2": "https://www.youtube.com/watch?v=b_eqnGY2_l0",
"Most expensive assembly table": "https://www.youtube.com/watch?v=4jsTo6WQ9Hc",
"I have been doing it all wrong": "https://www.youtube.com/watch?v=v_HVA391T0E",
"I finally found the perfect wood finish": "https://www.youtube.com/watch?v=3fvq4pQ6nhg",
"Kreg Foreman Electric Pocket Hole Machine - Is it right for you?": "https://www.youtube.com/watch?v=-e5MnK7h-i4",
"How To Install Cabinet Door Hinges": "https://www.youtube.com/watch?v=K2M0yn-y7J8",
"Installing Full Extension Drawer Slides": "https://www.youtube.com/watch?v=OGuoxKxCOV0",
"Install Cabinet Hardware Fast and Easy - True Position Tools Cabinet Hardware Jig": "https://www.youtube.com/watch?v=ItsgvmiL5-c",
"How to install Hewn Stoneform flooring": "https://www.youtube.com/watch?v=mL6YVmF908I",
"How To Make A Template For Woodworking": "https://www.youtube.com/watch?v=swYNcS_WBB4",
"I Built a Wall in my Garage": "https://www.youtube.com/watch?v=GGW7nyNWsPk",
"How To Install C Channels / Woodworking Tips": "https://www.youtube.com/watch?v=DOwaCkOOon8",
"How To Make Castle Joints": "https://www.youtube.com/watch?v=JFRnc49dpMs",
"How To Taper Legs on a Jointer": "https://www.youtube.com/watch?v=n2JINqaX5xk",
"How To Make Drawers": "https://www.youtube.com/watch?v=Oab0zq93LkI",
"How To Make Furniture Level // Woodworking Tips": "https://www.youtube.com/watch?v=LQxx7re5a4c",
"How To Make Shaker Doors": "https://www.youtube.com/watch?v=7-dT8RgYSRE",
"I would not buy these with your money": "https://www.youtube.com/watch?v=rklf9Dg6EaM",
"I Bet Your Forstner Bits Donot Do This": "https://www.youtube.com/watch?v=AMiCviHg9qQ",
"How to Install Lifeproof Vinyl Flooring": "https://www.youtube.com/watch?v=JMf2WXO15rg",
"How YOU Should Be Cutting Plywood": "https://www.youtube.com/watch?v=X0r_rkXtSrw",
"How To Install Blum Undermount Drawer Slides": "https://www.youtube.com/watch?v=K82l1ec7rR0",
"How To Make A Box Joint Jig": "https://www.youtube.com/watch?v=EXaWQMVsxj0",
"How To Install Mr Cool DIY Series": "https://www.youtube.com/watch?v=eh5EeFfEtJM",
"How To Spray Finishes": "https://www.youtube.com/watch?v=wh6cldHCkLA",
"I sold my bandsaw for this - Hammer N4400 Assembly": "https://www.youtube.com/watch?v=HJRlPG_R-eU",
"I sold my SawStop for a slider // Hammer K3 Winner Assembly": "https://www.youtube.com/watch?v=nX6I1ODeEwM",
"How to build drawers with slides": "https://www.youtube.com/watch?v=_YGXlrykVtM",
"FINALLY! The sprayer I have been waiting for": "https://www.youtube.com/watch?v=T8BIpNzdh7M",
"Biscuit Joiner vs Festool Domino": "https://www.youtube.com/watch?v=7GTbnJrXlaI",
"Building a giant conference table without a table saw": "https://www.youtube.com/watch?v=oJONJ1HQgZ4",
"Dining Table Build [PREMIUM WALNUT]": "https://www.youtube.com/watch?v=hprQSH0KpzY",
"Basic Cabinet Making [Compilation]": "https://www.youtube.com/watch?v=Oeu7ogH2NZU",
"Avoid this common dust collection mistake": "https://www.youtube.com/watch?v=tVgeQtN4HkA",
"Hammer A3-26 Jointer/Planer Combo": "https://www.youtube.com/watch?v=FJ3Y8yM0GjE",
"Grow Your Woodworking Business with Social Media / GA Woodworkers Guild Talk": "https://www.youtube.com/watch?v=PVWLtNApaHk",
"After Just 2 Years.........THIS HAPPENED!": "https://www.youtube.com/watch?v=LGrXxhx-OuM",
"Festool LR 32 - Intro": "https://www.youtube.com/watch?v=EO62T1LHdNA",
"DRILLNADO - A Must For Your Drill Press // TOOL RECOMMENDATION": "https://www.youtube.com/watch?v=fiilDxaeEWQ",
"Hammer A3-41 Assembly": "https://www.youtube.com/watch?v=k0JAZ9kBy6k",
"CNC Basics - Make Your First Cut": "https://www.youtube.com/watch?v=_Rt5sC9DrEQ",
"Festool for the newbie": "https://www.youtube.com/watch?v=mRa0XaTUGwU",
"How To Glue Up A Table Top": "https://www.youtube.com/watch?v=DOfqmYzs_EE",
"Finish Sprayer Comparison - Graco, Rockler, Homeright": "https://www.youtube.com/watch?v=To2ynH0w4kY",
"Festool Domino DF 500 Review and Overview": "https://www.youtube.com/watch?v=yqoruVpkVO0",
"Beginners guide to face frame cabinets": "https://www.youtube.com/watch?v=A33qbcPQ0N0",
"Beginners guide to frameless upper cabinets": "https://www.youtube.com/watch?v=MzBmkRYb8tA",
"2020 Shop Tour": "https://www.youtube.com/watch?v=R14XHp6Wuh0",
"Cabinet Face Frames - Easy": "https://www.youtube.com/watch?v=47W_Fl3qkG0",
"A genius trick no one shows": "https://www.youtube.com/watch?v=w7l1_d2GtuA",
"Complete stair makeover": "https://www.youtube.com/watch?v=2rkFnsTztog",
"Does Rubio Monocoat Hold Up Well?": "https://www.youtube.com/watch?v=n6ZUVhG7soE",
"Are corded tools obsolete?": "https://www.youtube.com/watch?v=8IF1bIWD4ws",
"15 Woodworking Tools You Will not Regret": "https://www.youtube.com/watch?v=Q1Fo9BRHB8I",
"Complete Mr Cool Install": "https://www.youtube.com/watch?v=Lrl40LRVlnc",
"10 Strategies for a More Efficient Wood Shop Layout": "https://www.youtube.com/watch?v=ORBWwocQOXI",
"10 woodworking tools I regret not buying sooner": "https://www.youtube.com/watch?v=IvyGUy75-YQ",
"From Concept to Reality: Building the Perfect Kitchen Island!": "https://www.youtube.com/watch?v=R5YYrt75VwM",
"American Green Lights": "https://www.youtube.com/watch?v=eHUK8GW-28s",
"10 Tools Every Woodworker Should Own": "https://www.youtube.com/watch?v=2vclE1hXc3A",
"Apply Rubio Monocoat Oil+2C I Bents Woodworking": "https://www.youtube.com/watch?v=wyAnr53BHu4",
"Festool TS 55 F First Look": "https://www.youtube.com/watch?v=PoRV4jd4lAc",
"Beginners guide to face frame doors and drawers": "https://www.youtube.com/watch?v=na5MJZVS80g",
"Carbon method 9 months later": "https://www.youtube.com/watch?v=a0oARF_W5uk",
"Cleaning A Saw Blade": "https://www.youtube.com/watch?v=MVOlISN1W_Q",
"5 Ways To Fill Knots And Imperfections // WOODWORKING TIPS": "https://www.youtube.com/watch?v=ync1AVpO26w",
"Every track saw owner could use this": "https://www.youtube.com/watch?v=imlY61oONlY",
"Festool Sander Comparison": "https://www.youtube.com/watch?v=J_BxZ48FbP0",
"25 tools I regret not buying sooner": "https://www.youtube.com/watch?v=8o9rJo5zvF4",
"Harvey G700 Gyro Air Dust Processer - 6 Months Later": "https://www.youtube.com/watch?v=bxHXBPuokP4",
"8 Tools I Regret Not Buying Sooner": "https://www.youtube.com/watch?v=iu02blZFgjI",
"A sliding table saw that fits in a box?": "https://www.youtube.com/watch?v=dA3zW7GVJaA",
"5 Woodworking Joints For Beginners": "https://www.youtube.com/watch?v=snjE9TF24dY",
"Better than the lamello connectors?": "https://www.youtube.com/watch?v=xBXWJWhQYQs",
"12 Tools I will Never REGRET Buying": "https://www.youtube.com/watch?v=Ypa0QL_lIKo",
"11 woodworking tools you need to own": "https://www.youtube.com/watch?v=CzcLWb8YbZk",
"10 Woodworking tools you will not regret": "https://www.youtube.com/watch?v=_T9jYc2rXuI",
"15 cabinet tools I do not regret": "https://www.youtube.com/watch?v=TuFCMLoFT9s",
"How I Finish Cutting Boards": "https://www.youtube.com/watch?v=529lkCNImY8",
"How To Apply Edge Banding // WOODWORKING TIPS": "https://www.youtube.com/watch?v=Sfp8wLcALN0",
"5 Tools to Buy EARLY // WOODWORKING TOOLS": "https://www.youtube.com/watch?v=ziNVWrYn-7Q",
"4 tips for applying edge banding": "https://www.youtube.com/watch?v=rg6b72zPxqw",
"5 Different Finishes, 5 Different Results [Walnut Edition]": "https://www.youtube.com/watch?v=i6K5zkiH_VM",
"5 Modifications I Made In My Garage Shop - New Shop Part 5": "https://www.youtube.com/watch?v=6_rSY8QYvb8",
}

# List of example questions
EXAMPLE_QUESTIONS = [
    "How do TSO Products' Festool accessories improve woodworking precision?",
    "What makes Bits and Bits Company's router bits ideal for woodworking?",
    "What are the benefits of Japanese saws and chisels from Taylor Toolworks?",
    "How does the Festool LR 32 System aid in cabinet making?",
    "What advantages does the Festool Trigger Clamp offer for quick release and one-handed use?"
]

def generate_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def add_product(title, tags, link):
    product_id = str(uuid.uuid4())
    tags_text = ', '.join(tags)
    embedding = generate_embedding(tags_text)
    
    metadata = {
        "title": title,
        "tags": tags_text,
        "link": link
    }
    
    product_index.upsert([(product_id, embedding, metadata)])
    return product_id

def get_all_products():
    # Get the total number of vectors in the index
    total_vectors = product_index.describe_index_stats()['total_vector_count']
    
    if total_vectors == 0:
        return []  # Return an empty list if there are no products
    
    # Fetch all vector IDs
    fetch_response = product_index.query(
        vector=[0] * 1536,  # Dummy vector
        top_k=total_vectors,
        include_metadata=True
    )
    
    products = []
    for match in fetch_response['matches']:
        metadata = match['metadata']
        products.append((match['id'], metadata['title'], metadata['tags'], metadata['link']))
    
    return products

def query_products_for_keywords(keywords):
    query_text = ', '.join(keywords)
    query_embedding = generate_embedding(query_text)
    
    results = product_index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    return [(match['id'], match['metadata']['title'], match['metadata']['tags'], match['metadata']['link']) 
            for match in results['matches']]

def delete_product(product_id):
    product_index.delete(ids=[product_id])

def update_product(product_id, title, tags, link):
    tags_text = ', '.join(tags)
    embedding = generate_embedding(tags_text)
    
    metadata = {
        "title": title,
        "tags": tags_text,
        "link": link
    }
    
    product_index.upsert([(product_id, embedding, metadata)])

def get_product_by_id(product_id):
    fetch_response = product_index.fetch(ids=[product_id])
    if product_id in fetch_response['vectors']:
        vector = fetch_response['vectors'][product_id]
        metadata = vector['metadata']
        return (product_id, metadata['title'], metadata['tags'], metadata['link'])
    return None

def load_initial_data():
    initial_products = [
        ("TSO Products", "Aftermarket Festool accessories, Precision woodworking tools, Router table inserts, Guide rail accessories, Dust collection adapters", "https://tsoproducts.com/?aff=5"),
        ("Bits and Bits Company", "Router bits, Drill bits, Saw blades, Woodworking accessories, Carbide cutting tools", "http://bit.ly/bitsbitsbw"),
        ("Taylor Toolworks", "Woodworking hand tools, Japanese saws, Chisels, Sharpening supplies, Layout tools", "https://lddy.no/1e5hv"),
        ("Festool LR 32 System", "Cabinet making, Shelf pin holes, Precision drilling, 32mm system, Modular shelving", "https://amzn.to/3hRTvLB"),
        ("Festool Trigger Clamp", "Quick release, One-handed operation, Versatile clamping, Woodworking, Assembly, Glue-ups", "https://amzn.to/2HoVydC")
    ]
    
    for title, tags, link in initial_products:
        add_product(title, tags.split(', '), link)
    
    st.success("Initial product data loaded successfully!")

def safe_run_tree(name, run_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with trace(name=name, run_type=run_type, client=langsmith_client) as run:
                    result = func(*args, **kwargs)
                    run.end(outputs={"result": str(result)})
                    return result
            except Exception as e:
                st.error(f"Error in LangSmith tracing: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_metadata_from_text(text):
    title = text.split('\n')[0] if text else "Untitled Video"
    return {"title": title}

def upsert_transcript(transcript_text, metadata):
    chunks = [transcript_text[i:i+8000] for i in range(0, len(transcript_text), 8000)]
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        chunk_metadata = metadata.copy()
        chunk_metadata['text'] = chunk
        chunk_metadata['chunk_id'] = f"{metadata['title']}_chunk_{i}"
        transcript_index.upsert([(chunk_metadata['chunk_id'], embedding, chunk_metadata)])

def query_transcripts(query):
    query_embedding = generate_embedding(query)
    result = transcript_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return [(match['metadata']['title'], match['metadata']['text']) for match in result['matches']]

@safe_run_tree(name="generate_keywords", run_type="llm")
def generate_keywords(text):
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    system_message = SystemMessage(content="You are a specialized keyword extraction system for woodworking terminology. Extract 3-5 highly relevant and specific keywords or short phrases from the given text, focusing on technical terms, tool names, or specific woodworking techniques.")
    human_message = HumanMessage(content=f"Generate keywords from this text: {text}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    keywords = response.content.strip().split(',')
    return [keyword.strip().lower() for keyword in keywords if keyword.strip()]

@safe_run_tree(name="get_answer", run_type="chain")
def get_answer(context, user_query):
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    system_message = SystemMessage(content="You are Jason Bent's woodworking expertise embodied in an AI. Answer the user's query based on the provided context, incorporating relevant product information without mentioning specific product names.")
    human_message = HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    initial_answer = response.content
    
    query_keywords = generate_keywords(user_query)
    answer_keywords = generate_keywords(initial_answer)
    all_keywords = list(set(query_keywords + answer_keywords))
    related_products = query_products_for_keywords(all_keywords)
    
    system_message_2 = SystemMessage(content="Refine the answer to incorporate product information without naming specific products. Ensure the response is comprehensive, reflects Jason's expertise, and includes specific techniques or advice.")
    human_message_2 = HumanMessage(content=f"Initial Answer: {initial_answer}\n\nRelated Products: {related_products}\n\nProvide a final answer.")
    
    with get_openai_callback() as cb:
        final_response = chat([system_message_2, human_message_2])
    final_answer = final_response.content
    
    return final_answer, related_products, all_keywords

def process_query(query):
    if query:
        with st.spinner("Searching for the best answer..."):
            matches = query_transcripts(query)
            if matches:
                context = " ".join([f"Title: {title}\n{text}" for title, text in matches])
                final_answer, related_products, keywords = get_answer(context, query)
                
                st.write(final_answer)
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                col1, space, col2 = st.columns([0.47, 0.06, 0.47])
                
                with col1:
                    st.subheader("Related Video")
                    video_displayed = False
                    for title, _ in matches:
                        if title in YOUTUBE_LINKS:
                            video_id = YOUTUBE_LINKS[title].split("v=")[1].split("&")[0]
                            st.markdown(f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)
                            st.caption(f"Video: {title}")
                            video_displayed = True
                            break
                    if not video_displayed:
                        st.write("No related video found.")
                
                with col2:
                    st.subheader("Related Products")
                    if related_products:
                        for product in related_products:
                            st.markdown(f"[{product[1]}]({product[3]})")
                    else:
                        st.write("No related products found.")
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((query, final_answer, related_products, matches[0][0] if matches else None))
            else:
                st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
    else:
        st.warning("Please enter a question before searching.")

def query_interface():
    st.header("Woodworking Assistant")

    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    st.subheader("Popular Questions")
    for question in st.session_state.selected_questions:
        if st.button(question, key=question):
            st.session_state.current_query = question

    user_query = st.text_input("What would you like to know about woodworking?")
    if st.button("Get Answer"):
        st.session_state.current_query = user_query

    if st.session_state.current_query:
        st.subheader("Response:")
        process_query(st.session_state.current_query)
        st.session_state.current_query = ""

    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("Recent Questions and Answers")
        for i, (q, a, products, video_title) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {q}"):
                st.write(f"A: {a}")
                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.subheader("Related Video")
                    if video_title and video_title in YOUTUBE_LINKS:
                        video_id = YOUTUBE_LINKS[video_title].split("v=")[1].split("&")[0]
                        st.markdown(f'<iframe width="100%" height="215" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)
                        st.caption(f"Video: {video_title}")
                    else:
                        st.write("No related video found.")
                with col2:
                    st.subheader("Related Products")
                    if products:
                        for product in products:
                            st.markdown(f"[{product[1]}]({product[3]})")
                    else:
                        st.write("No related products found.")

def database_interface():
    st.subheader("All Products")
    products = get_all_products()
    if products:
        df = pd.DataFrame(products, columns=['ID', 'Title', 'Tags', 'Links'])
        st.dataframe(df)
    else:
        st.write("The database is empty.")
        if st.button("Load Initial Data"):
            load_initial_data()

    st.subheader("Add New Product")
    new_title = st.text_input("Title")
    new_tags = st.text_input("Tags (comma-separated)")
    new_link = st.text_input("Link")
    if st.button("Add Product"):
        if new_title and new_tags and new_link:
            product_id = add_product(new_title, new_tags.split(','), new_link)
            st.success(f"Product added successfully with ID: {product_id}")
        else:
            st.warning("Please fill in all fields.")

    st.subheader("Update Product")
    update_id = st.text_input("Enter Product ID to update")
    if update_id:
        product = get_product_by_id(update_id)
        if product:
            update_title = st.text_input("New Title", value=product[1])
            update_tags = st.text_input("New Tags", value=product[2])
            update_link = st.text_input("New Link", value=product[3])
            if st.button("Update Product"):
                update_product(update_id, update_title, update_tags.split(','), update_link)
                st.success(f"Product with ID {update_id} updated successfully!")
        else:
            st.warning(f"No product found with ID {update_id}")

    st.subheader("Delete Product")
    delete_id = st.text_input("Enter Product ID to delete")
    if st.button("Delete Product"):
        delete_product(delete_id)
        st.success(f"Product with ID {delete_id} deleted successfully!")

def main():
    st.set_page_config(page_title="Bent's Woodworking Assistant", layout="wide")

    if not LANGCHAIN_API_KEY:
        st.warning("LangSmith API key is not set. Some features may not work properly.")

    st.image("bents logo.png", width=150)
    st.title("Bent's Woodworking Assistant")

    # Check if the database is empty and offer to load initial data
    if not get_all_products():
        st.warning("The product database is empty. Would you like to load some initial data?")
        if st.button("Load Initial Data"):
            load_initial_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Query Interface", "Database Management"])

    if page == "Query Interface":
        query_interface()
    elif page == "Database Management":
        database_interface()

    if page == "Query Interface":
        with st.sidebar:
            st.header("Upload Transcripts")
            uploaded_files = st.file_uploader("Upload YouTube Video Transcripts (DOCX)", type="docx", accept_multiple_files=True)
            if uploaded_files:
                all_metadata = []
                for uploaded_file in uploaded_files:
                    transcript_text = extract_text_from_docx(uploaded_file)
                    metadata = extract_metadata_from_text(transcript_text)
                    all_metadata.append((metadata, transcript_text))
                st.subheader("Uploaded Transcripts")
                for metadata, _ in all_metadata:
                    st.text(f"Title: {metadata['title']}")
                if st.button("Upsert All Transcripts"):
                    with st.spinner("Upserting transcripts..."):
                        for metadata, transcript_text in all_metadata:
                            upsert_transcript(transcript_text, metadata)
                        st.success("All transcripts upserted successfully!")

if __name__ == "__main__":
    main()
