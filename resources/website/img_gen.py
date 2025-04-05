import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
import warnings
from io import BytesIO
import json
from stability_sdk import client 
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation 


def setup_dalle(apikey):
    """
    Sets up the OpenAI API for DALL-E image generation.
    
    Parameters:
    api_key (str): Your OpenAI API key.
    """
    openai_client = OpenAI(api_key=apikey)
    
    print("DALL-E setup complete.")
    return openai_client

def create_dalle_image(api_client,prompt,dimensions: tuple):
    """
    Generates an image from a given prompt using DALL-E.
    
    Parameters:
    prompt (str): The prompt to generate the image from.
    
    Returns:
    str: URL of the generated image.
    """
    response = api_client.images.generate(
            model="dall-e-3",
            prompt=prompt,   
            n=1,    
            quality="standard"    
            )  
   
    image_url = response.data[0].url
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img
