# **Project 3 Team 2 - Mystic AI**

# AI Story Creator & Adventure Generator

  

## Project Overview

### Mystic AI - Interactive Storybook Application

Mystic AI is a Streamlit-based web application that creates an interactive storybook experience using OpenAI's ChatGPT and DALL-E. Users can input a story genre or theme, and the app generates a dynamic story with accompanying images and user choices.

---

### Features
1. **API Key Authentication**:
   - Users must provide their OpenAI API key to access the app's features.
   - The key is securely stored in the session state.

2. **Dynamic Story Generation**:
   - Stories are generated using OpenAI's ChatGPT based on the user's input genre or theme.
   - Each story section includes text, user choices, and optionally an AI-generated image.

3. **AI-Generated Images**:
   - DALL-E generates images based on prompts extracted from the story.

4. **Interactive User Choices**:
   - Users can make choices at each story section, influencing the next part of the story.

5. **Session State Management**:
   - The app uses Streamlit's session state to manage user inputs, story sections, and app state.

6. **Sidebar Configuration**:
   - Includes instructions, API key input, and app information.  
---

### Purpose of Use

The tool is designed for:

- Interactive storytelling

- Educational purposes

- AI-assisted narrative generation
---

### File Navigation

-  `resources/website/app.py`: Entry point for the application

-  `resources/website/llm_init.py`: OpenAI LLM initialization and prompt handling

-  `resources/website/img_gen.py`: DALL-E 3 Image generation helper

-  `resources/website/requirements.txt`: dependencies for the UI

-  `resources/website/icons`: images for the UI

-  `config.env`: Environment variables for API keys

-  `README.md`: This documentation
---

### How It Works
1. **Setup**:
   - Users enter their OpenAI API key in the sidebar to authenticate.
   - They input a story genre or theme to begin.

2. **Story Generation**:
   - The app sends the user input to ChatGPT to generate a story.
   - Prompts for images are extracted and sent to DALL-E for image generation.

3. **Interactive Storytelling**:
   - Each story section is displayed with text, an image, and user choices.
   - User selections determine the next part of the story.

4. **Dynamic Updates**:
   - The app dynamically updates the story sections and maintains state across interactions.
---

### Key Components
1. **auth()**:
   - Authenticates the OpenAI API key and updates session state.

2. **get_story_and_image()**:
   - Generates story text, user options, and an AI-generated image.

3. **get_output()**:
   - Handles user input and updates the story dynamically.

4. **generate_content()**:
   - Renders each story section with text, choices, and images.

5. **add_new_data()**:
   - Adds new story sections to the session state.

6. **Streamlit Widgets**:
   - Sidebar for API key input and instructions.
   - Main container for story input, clear/reset buttons, and story rendering.
---

### Installation

1. Clone the repository:

```bash

git clone https://github.com/xraySMULu/project-3

```

2. cd resources/website/app.py

2. Install dependencies:

```bash

pip install -r requirements.txt

```

### Usage Instructions
1. Run the main script:
	

```bash

streamlit run resources/website/app.py

```
2. Enter your OpenAI API key in the sidebar.
3. Input a story genre or theme in the main container.
4. Click "Begin story" to start generating the story.
5. Make choices at each story section to progress the story.
6. Enjoy the interactive storytelling experience!
 ---

### Demo & Slideshow

Slide deck in PDF can be found in the `resources\presentation` folder.

---  

### Data Pre-processing and Gathering Steps

- Extract text from a provided PDF or text file

- Segment the story into logical scenes

- Identify choice points and generate alternatives
---  
  

### Visuals and Explanations

- Flowcharts depicting the branching logic

- Screenshots of the generated story paths
---  
  

### Additional Explanations and Major Findings

- The model effectively creates diverse branching paths

- Some limitations in complex story logic may require human refinement
---  
  

### Additional Questions That Surfaced and Plan for Future Development

- How can we improve character consistency in AI-generated choices?

- Can we integrate a UI for better user interaction?

- Future updates may include support for voice input and game export options

  

### Conclusion and References

This project demonstrates the power of AI in transforming static stories into interactive experiences.

  

### References

- [LangChain Documentation](https://python.langchain.com/)

- [OpenAI API](https://platform.openai.com/docs/)

- [Pinecone Documentation](https://www.pinecone.io/)