# **Project 3 Team 2 - Mystic AI**
# AI Story Creator & Adventure Generator

## Project Overview
This project is an AI-powered tool designed to parse a story and generate a dynamic "Choose Your Adventure" experience. By leveraging Large Language Models (LLMs), the tool enables users to input a genre, which is then processed to create an interactive adventure narrative. 

## File Navigation
- `resources/website/app.py`: Entry point for the application
- `resources/website/llm_init.py`: OpenAI LLM initialization and prompt handling
- `resources/website/img_gen.py`: DALL-E 3 Image generation helper
- `resources/website/requirements.txt`: dependencies for the UI
- `resources/website/icons`: images for the UI
- `config.env`: Environment variables for API keys
- `README.md`: This documentation

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

## Usage
1. Run the main script:
   ```bash
   streamlit run resources/website/app.py
   ```
2. In the sidebar, enter your OpenAI API key.
3. After entering the API key, please enter the genre/theme of your desired story. 
4. The AI will parse the genre, execute a prompt and direct the user through an interactive experience.

## Demo & Slideshow
Screenshots and walkthrough videos can be found in the `docs/demo` folder.

## Purpose of Use
The tool is designed for:
- Interactive storytelling
- Educational purposes
- AI-assisted narrative generation

## Data Pre-processing and Gathering Steps
- Extract text from a provided PDF or text file
- Segment the story into logical scenes
- Identify choice points and generate alternatives

## Visuals and Explanations
- Flowcharts depicting the branching logic
- Screenshots of the generated story paths

## Additional Explanations and Major Findings
- The model effectively creates diverse branching paths
- Some limitations in complex story logic may require human refinement

## Additional Questions That Surfaced and Plan for Future Development
- How can we improve character consistency in AI-generated choices?
- Can we integrate a UI for better user interaction?
- Future updates may include support for voice input and game export options

## Conclusion and References
This project demonstrates the power of AI in transforming static stories into interactive experiences. 

### References
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Pinecone Documentation](https://www.pinecone.io/)
