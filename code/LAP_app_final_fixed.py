import uuid
import os
import streamlit as st
from PIL import Image
from time import sleep
from streamlit_extras.app_logo import add_logo
from streamlit.components.v1 import html
from streamlit.delta_generator import DeltaGenerator

from llm_init import initialize_model  # Custom module for initializing the language model
from img_gen import *  # Custom module for image generation

# Function to authenticate API key and update session states
def auth():    
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key  # Set OpenAI API key
    st.session_state.genreBox_state = False  # Disable genre input box
    st.session_state.apiBox_state = True    # Enable API input box


# Configure the Streamlit page (title, icon, layout, and menu)
st.set_page_config(
    page_title='Mystic AI',  # Set page title
    page_icon=Image.open('icons/mai.ico'),  # Set page icon
    layout='wide',  # Use wide layout
    menu_items={
        'About': "Mystic AI is a dynamic story creator built with LangChain, OpenAI and DALL-E 3"  # About section
    },
    initial_sidebar_state='expanded'  # Expand sidebar by default
)

# Set the main title of the app
st.title(f"Mystic AI")  # Display app title

# Initialize session state variables if not already set

if 'user_choices' not in st.session_state:
    st.session_state['user_choices'] = 0
if 'story_ended' not in st.session_state:
    st.session_state['story_ended'] = False
if 'selections' not in st.session_state:
    st.session_state['selections'] = []

if 'cols' not in st.session_state:  # Stores story sections
    st.session_state['cols'] = []
if 'keep_graphics' not in st.session_state:  # Toggle for keeping graphics
    st.session_state['keep_graphics'] = False
if 'data_dict' not in st.session_state:  # Stores story data
    st.session_state['data_dict'] = {}
if 'genreBox_state' not in st.session_state:  # Controls genre input box visibility
    st.session_state['genreBox_state'] = True
if 'apiBox_state' not in st.session_state:  # Controls API input box visibility
    st.session_state['apiBox_state'] = False
if 'openai_api_key' not in st.session_state:  # Stores OpenAI API key
    st.session_state['openai_api_key'] = ''
if 'genre_input' not in st.session_state:  # Stores user input for story genre
    st.session_state['genre_input'] = 'Use a random theme'

# Configuring the Sidebar
with st.sidebar:
    st.image('icons/mysai.png')  # Display sidebar logo

    st.markdown('''
    Mystic AI is a dynamic story creator built with LangChain, OpenAI and DALL-E.
    ''')  # Sidebar description
    
    with st.expander('Instructions'):  # Expandable instructions section
        st.markdown('''
        - To begin Mystic AI, please enter your own OpenAI API key.
        - After entering the API keys, please enter the genre/theme of your desired story, and watch the magic unfold.
        ''')  # Instructions for using the app
    
    # Sidebar Form, wherein the user enters their API Keys. 
    with st.form(key='API Keys'):
        openai_key = st.text_input(
            label='Your OpenAI API Key', 
            key='openai_api_key',
            type='password',
            disabled=st.session_state.apiBox_state,
            help='You can create your own OpenAI API key by going to https://platform.openai.com/account/api-keys (Sign up required)'
        )       
        
        btn = st.form_submit_button(label='Begin Mystic AI!', on_click=auth)

    
    st.info('**Note:** You can close the sidebar when you enter the API keys')

# Displaying the API Key warnings
if not openai_key.startswith('sk-'): 
    st.warning('Please enter your OpenAI API Key to start Mystic AI', icon='⚠')


# Defining the functions for the actual screen
def get_story_and_image(user_resp):
    # Initialize the language model and DALL-E client
    llm_model = initialize_model()
    openai_client = setup_dalle(st.session_state.openai_api_key)
        
    # Generate a response from the language model
    bot_response = llm_model.predict(input=user_resp)
    print(bot_response)
    response_list = bot_response.split("\n")
    
    # Filter out empty lines and separators from the response
    responses = list(filter(lambda x: x != '' and x != '-- -- --', response_list))
    
    # Generate an image using DALL-E if the response contains an image prompt
    if len(response_list) != 1:
        img_prompt = response_list[-1]
        dalle_img = create_dalle_image(openai_client, img_prompt)        
    else:
        dalle_img = None
        
    # Remove unwanted lines related to image generation from the response
    responses = list(filter(lambda x: 'DALL-E' not in x and 'Image prompt' not in x, responses))
    
    # Parse the response into story, label, and options
    opts = []
    story = ''
    label = ''
    
    for response in responses:
        response = response.strip()  
        
        try:
            if response.startswith(('What', 'Which', 'Choose')):
                label = f'**{response}**'
            elif response[1:2] in {'.', ')'} or response[1:6] == ' --' or response.startswith('Option'):
                opts.append(response)
            else:
                story += f'{response}\n'
        except IndexError:
            print(f"IndexError: The response '{response}' is too short to check the specified index.")
            st.switch_page('app.py')  # Navigate to an error page if the response is too short


        # if response.startswith('What') or response.startswith('Which') or response.startswith('Choose'):
        #     label = '**' + response + '**'
        # elif response[1] == '.' or response[1] == ')' or response[1:6] == ' --' or response.startswith('Option'):
        #     opts.append(response) 
        # else:
        #     story += response + '\n'  # Append to the story text    
    
    # Return the parsed story, label, options, and generated image
    if not story:
        story = 'This fantasy story is about embarking on an epic quest. The hero must make crucial decisions that will shape their destiny and the fate of the realm.'  # Default story if none is provided

    if not label:
        label = '**What will you choose?**'
    
    # Check if the number of options exceeds 6
    if len(opts) > 6:        
        opts = trim_lst(opts)  # Validate the options list
    elif len(opts) < 6:  # If less than 6 options
        # Ensure the options list contains all required values
        opts = ensure_lst_values(opts)

    return {
        'Story': story,
        'Radio Label': label,
        'Options': opts,
        'Image': dalle_img
    }


def trim_lst(lst):
    
    # Initialize an empty list to store the ordered options
    opts_rtn = []
    opts_ordered_1 = []
    opts_ordered_2 = []
       
    # Iterate through the responses and keep only the first 6 values corresponding to A) to F)
    for op in lst:
        if op.startswith(('A)', 'B)', 'C)', 'D)', 'E)', 'F)')):
            opts_ordered_1.append(op)
        if len(opts_ordered_1) == 6:
            break
        elif op.startswith(('A.', 'B.', 'C.', 'D.', 'E.', 'F.')):
            opts_ordered_2.append(op)
        if len(opts_ordered_2) == 6:
            break    
    # Sort the options based on the desired order
    opts_rtn = sort_lst(opts_ordered_1, opts_ordered_2)

    return opts_rtn

def ensure_lst_values(lst):
    required_values = ['A', 'B', 'C', 'D', 'E', 'F']
    default_value = "Think about what you want to do next."

    # Create a set of the first characters in the list
    existing_values = {item[0] for item in lst}

    # Check for missing values and add them with the default value
    for value in required_values:
        if value not in existing_values:
            lst.append(f"{value}) {default_value}")
    
    # Sort the list based on the desired order
    lst = sort_lst_by_char1(lst)
    # Return the updated list
    return lst

def sort_lst_by_char1(lst):
    lst_rtn = []
    # Define the desired order
    desired_order = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Filter and sort the ops based on the desired order
    opts1 = sorted(
        [op for op in lst if op[:1] in desired_order],
        key=lambda x: desired_order.index(x[:1])
    )

    if len(opts1) == 6:
        lst_rtn = opts1
    
    return lst_rtn

def sort_lst(lst1,lst2):
    lst_rtn = []
    # Define the desired order
    desired_order_1 = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
    desired_order_2 = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']

    # Filter and sort the ops based on the desired order
    opts1 = sorted(
        [op for op in lst1 if op[:2] in desired_order_1],
        key=lambda x: desired_order_1.index(x[:2])
    )       
    opts2 = sorted(
        [op for op in lst2 if op[:2] in desired_order_2],
        key=lambda x: desired_order_2.index(x[:2])
    )

    if len(opts1) == 6:
        lst_rtn = opts1
    elif len(opts2) == 6:   
        lst_rtn = opts2
    return lst_rtn

# Function to handle user input, generate story content, and update session state


def get_output(_user_choice):
    if st.session_state.get('story_ended', False) or st.session_state.get('user_choices', 0) >= 5:
        return

    st.session_state.keep_graphics = True
    st.session_state['user_choices'] += 1
    st.session_state['selections'].append(_user_choice)

    data = get_story_and_image(_user_choice)
    add_new_data(data['Story'], data['Radio Label'], data['Options'], data['Image'])

        user_choice = str(st.session_state[f'radio_{el_id}'])
    
    if genre:         
        st.session_state['genreBox_state'] = False
        user_choice = genre
    
    with _pos:    
        data = get_story_and_image(user_choice)
        add_new_data(data['Story'], data['Radio Label'], data['Options'], data['Image'])
    
    
def generate_content(story, lbl_text, opts: list, img, el_id):   
    if f'expanded_{el_id}' not in st.session_state:
        st.session_state[f'expanded_{el_id}'] = True
    if f'radio_{el_id}_disabled' not in st.session_state:
        st.session_state[f'radio_{el_id}_disabled'] = False
    if f'submit_{el_id}_disabled' not in st.session_state:
        st.session_state[f'submit_{el_id}_disabled'] = False
    
    story_pt = list(st.session_state["data_dict"].keys()).index(el_id) + 1
    expander = st.expander(f'Part {story_pt}', expanded=st.session_state[f'expanded_{el_id}'])   
    col1, col2 = expander.columns([0.65, 0.35])
    empty = st.empty()
    if img:
        col2.image(img, width=40)
    
    with col1:
        st.write(story)        
        if lbl_text and opts:
            with st.form(key=f'user_choice_{el_id}'): 
                st.radio(lbl_text, opts, disabled=st.session_state[f'radio_{el_id}_disabled'], key=f'radio_{el_id}')
                st.form_submit_button(
                    label="Let's do this!", 
                    disabled=st.session_state[f'submit_{el_id}_disabled'], 
                    on_click=get_output, args=[empty], kwargs={'el_id': el_id}
                )


def add_new_data(*data):   
    el_id = str(uuid.uuid4())
    st.session_state['cols'].append(el_id)
    st.session_state['data_dict'][el_id] = data
    

# Genre Input widgets
with st.container():
    # Create three columns for input, clear button, and begin button
    col_1, col_2, col_3 = st.columns([8, 1, 1], gap='small')
    
    # Text input for entering the story theme/genre
    col_1.text_input(
        label='Enter the theme/genre of your story',
        key='genre_input',
        placeholder='Enter the theme/genre of the story', 
        disabled=st.session_state.genreBox_state
    )
    
    # Clear button to reset the genre input field
    col_2.write('')
    col_2.write('')
    col_2_cols = col_2.columns([0.5, 6, 0.5])
    col_2_cols[1].button(
        ':arrows_counterclockwise: &nbsp; Clear', 
        key='clear_btn',
        on_click=lambda: setattr(st.session_state, "genre_input", ''),
        disabled=st.session_state.genreBox_state
    )
    
    # Begin button to start generating the story
    col_3.write('')
    col_3.write('')
    begin = col_3.button(
        'Begin story →',
        on_click=get_output, args=[st.empty()], kwargs={'genre': st.session_state.genre_input},
        disabled=st.session_state.genreBox_state
    )
    
# Generate content for each story section stored in session state
for col in st.session_state['cols']:
    data = st.session_state['data_dict'][col]
    generate_content(data[0], data[1], data[2], data[3], col)