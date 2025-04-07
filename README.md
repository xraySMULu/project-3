# **Mystic Ai: Dynamic Story Creator**
<a id="idtop"></a>  
<img src="./resources/content/hdr.png" width="750">

## Table of Contents
* [Project Overview](#overview)
* [Features](#features)
* [Purpose of Use](#purpose)
* [File Navigation](#filenav)
* [How It Works](#howit)
* [Key Components](#keycomp)
* [Installation](#installation)
* [Usage Instructions](#usage)
* [Demo and Slideshow](#demos)
* [Application Development](#appdev)
* [Additional Explanations](#addex)
* [Major Findings](#majfind)
* [Additional questions that surfaced](#addques)
* [Plan For Future Development](#plan)
* [Conclusion](#conclusion)
* [References](#references)
  

## Project Overview

## Mystic AI - Interactive Storybook Application

Mystic AI is a Streamlit-based web application that creates an interactive storybook experience using OpenAI's ChatGPT and DALL-E. Users can input a story genre or theme, and the app generates a dynamic story with accompanying images and user choices.

---

## Features
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

## Purpose of Use

The tool is designed for:

- Interactive storytelling

- Educational purposes

- AI-assisted narrative generation
---

## File Navigation
* Website
   -  [Resources/website](Resources/website) - Directory containing all of the website files used by the code
	-  `resources/website/app.py`: Entry point for the web application		
	-  `resources/website/llm_init.py`: OpenAI LLM initialization and prompt handling
	-  `resources/website/img_gen.py`: DALL-E 3 Image generation helper
	-  `resources/website/requirements.txt`: dependencies for the UI
	-  `resources/website/icons`: images for the UI

* Project
	-  [Resources/content](Resources/content) - Directory containing all of the image files used by the code
 	-  [Resources/data](Resources/data) - Directory containing all of the data files used by the code
 	-  [Resources/data](Resources/presentation) - Directory containing all of the presentation files used by the code 	
 	-  `README.md`: This documentation 	
 ---

## How It Works
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

## Key Components
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

## Installation

1. Clone the repository:

```bash

git clone https://github.com/xraySMULu/project-3

```

2. cd resources/website/app.py

2. Install dependencies:

```bash

pip install -r requirements.txt

```

## Usage Instructions
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

## Demo and Slideshow

**Mystic AI Demo**

* Mystic AI Demo

* Navigate to [Demo](resources/content/ui_demo.gif)

**Mystic AI - DALL-E Prints**

![image](resources/content/warrior1.png)
![image](resources/content/team1.png)
![image](resources/content/team2.png)
![image](resources/content/spook1.png)

**Slideshow**

* Project #3 - Team #2 - Slideshow

* Navigate to [Slideshow PDF](resources/presentation/proj3slideshow.pdf)

---  

## Application Development

For the Mystic AI application, the application development steps involved preparing model prompt, user inputs, managing API responses, and ensuring the generated content is suitable for display. Below is an outline of the data and preprocessing steps:

1. **User Input Handling**:
   - **Input**: Users provide a story genre or theme and their OpenAI API key.
   - **Validation**: 
     - Ensure the API key starts with `sk-` to confirm it is valid.
     - Sanitize the genre input to remove unnecessary whitespace or invalid characters.

2. **API Key Authentication**:
   - Store the OpenAI API key securely in the session state.
   - Set the key as an environment variable for use in API calls.

3. **Language Model Response Processing**:
   - **Input**: User-provided genre or story continuation is sent to OpenAI's ChatGPT.
   - **Response Parsing**:
     - Split the response into lines and filter out empty lines or separators (e.g., `-- -- --`).
     - Identify and separate the story text, user choice labels, and options.
     - Ensure options are limited to a maximum of 6 for better user experience.

4. **Image Prompt Extraction**:
   - Extract the image prompt from the ChatGPT response (if present).
   - Remove any lines related to image generation from the story text.

5. **Image Generation**:
   - Use the extracted image prompt to generate an image with DALL-E.
   - Validate the generated image:
     - Ensure the image meets the minimum size requirement (e.g., 256x256 pixels).
     - Resize the image if necessary to meet display requirements.

6. **Session State Management**:
   - Store the processed story, user options, and generated image in the session state.
   - Maintain a list of story sections (`cols`) to dynamically render content.

7. **Dynamic Content Rendering**:
   - Use the processed data to display the story, user choices, and images in the app.
   - Ensure user interactions (e.g., radio button selections) are captured and processed to generate the next part of the story.

8. **Error Handling**:
   - Handle API errors (e.g., invalid API key, rate limits) gracefully by displaying appropriate warnings.
   - Provide fallback behavior if image generation fails (e.g., display text-only content).

---

## Additional Explanations

1. **AI Integration**: Mystic AI combines OpenAI's ChatGPT for natural language processing and storytelling with DALL-E for generating visually appealing images. This integration demonstrates how AI can work cohesively to create an immersive user experience.

2. **User Interactivity**: The application allows users to actively participate in the storytelling process by selecting themes and making choices that influence the narrative. This interactivity highlights the potential of AI to create personalized and engaging content.

3. **Session State Management**: By leveraging Streamlit's session state, the app ensures a seamless user experience, maintaining story progression and user inputs across interactions without requiring page reloads.

4. **Error Handling**: The app includes mechanisms to validate user inputs (e.g., API key format) and handle potential API errors gracefully, ensuring reliability and usability.

5. **Scalability**: The modular design of the application allows for future enhancements, such as integrating additional AI models, expanding storytelling capabilities, or supporting more complex user interactions.

## Major Findings

1. **AI's Creative Potential**: The application demonstrates that AI can generate coherent, engaging, and contextually relevant stories and visuals, showcasing its potential in creative industries.

2. **User Engagement**: Allowing users to influence the narrative through choices significantly enhances engagement, making the storytelling experience more interactive and personalized.

3. **Seamless Integration**: The combination of ChatGPT and DALL-E highlights the effectiveness of integrating multiple AI models to deliver a unified experience.

4. **Importance of Preprocessing**: Properly parsing and cleaning AI-generated responses is critical to ensuring the content is user-friendly and visually appealing.

5. **Accessibility of AI**: By providing an intuitive interface, Mystic AI makes advanced AI technologies accessible to a broader audience, including those without technical expertise.

6. **Challenges in Image Generation**: While DALL-E generates impressive visuals, ensuring the images meet size and quality requirements can be a challenge, requiring additional preprocessing steps.

7. **Scalability and Adaptability**: The app's modular architecture allows for easy expansion, making it adaptable for various use cases, such as education, entertainment, or creative writing tools.

## Additional questions that surfaced

1.  **How can we improve the coherence of AI-generated stories?**
    
    -   While ChatGPT generates engaging narratives, ensuring logical consistency across multiple story sections remains a challenge. How can we refine prompts or use memory mechanisms to improve this?
2.  **What are the limits of user interactivity?**
    
    -   How many choices or branching paths can we realistically support before the complexity becomes unmanageable for both the user and the application?
3.  **How can we optimize image generation?**
    
    -   DALL-E occasionally produces images that do not align perfectly with the story context. How can we improve prompt engineering or incorporate user feedback to refine image generation?
4.  **What additional features would enhance user engagement?**
    
    -   Would features like saving stories, sharing them, or adding sound effects/music improve the overall experience?
5.  **How scalable is the application?**
    
    -   As more users interact with the app, how can we ensure that API rate limits, server performance, and caching mechanisms can handle increased demand?

## Plan for Future Development

1.  **Enhance Story Coherence**:
    
    -   Implement memory mechanisms or context-passing techniques to ensure consistency across story sections.
    -   Explore fine-tuning ChatGPT with custom datasets for more domain-specific storytelling.
2.  **Expand User Interactivity**:
    
    -   Add support for more complex branching narratives and user-defined story elements.
    -   Introduce a "sandbox mode" where users can directly edit or guide the story.
3.  **Improve Image Generation**:
    
    -   Develop better prompt engineering techniques for DALL-E to align visuals more closely with the story.
    -   Allow users to provide feedback on generated images and regenerate them if needed.
4.  **Introduce Story Saving and Sharing**:
    
    -   Enable users to save their stories locally or in the cloud.
    -   Add options to share stories on social media or export them as PDFs.
5.  **Optimize Performance**:
    
    -   Implement caching mechanisms for frequently used API calls to reduce latency.
    -   Explore server-side deployment options to handle higher user loads.
6.  **Incorporate Additional AI Models**:
    
    -   Experiment with other AI models for storytelling or image generation, such as Stable Diffusion or GPT-4.
    -   Add support for voice synthesis to narrate the story.
7.  **Expand Use Cases**:
    
    -   Adapt the application for educational purposes, such as teaching creative writing or history through interactive stories.
    -   Explore gamification elements to make the experience more engaging.
8.  **User Feedback Integration**:
    
    -   Collect user feedback to identify pain points and prioritize new features.
    -   Conduct usability testing to refine the interface and improve the overall experience.

By addressing these questions and implementing these plans, Mystic AI can evolve into a more robust, scalable, and engaging platform that continues to push the boundaries of AI-driven storytelling.

## Conclusion

This project demonstrates the power of AI in transforming static stories into interactive experiences.



## References

- [LangChain Documentation](https://python.langchain.com/)

- [OpenAI API](https://platform.openai.com/docs/)

- [DALL-E 3 API](https://help.openai.com/en/articles/8555480-dall-e-3-api)
