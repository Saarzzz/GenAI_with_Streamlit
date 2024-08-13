import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set the page configuration to wide mode
st.set_page_config(layout="wide",page_icon="✍️",page_title="Article Generator✍️")

# Function to save generated articles to a file
def save_article_to_file(topic, article, filename='article_history.txt'):
    with open(filename, 'a') as file:
        file.write(f"Topic: {topic}\n{article}\n\n\n\n\n")

# Function to load saved articles from a file
def load_articles_from_file(filename='article_history.txt'):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as file:
        articles = file.read().strip().split('\n\n\n\n\n')
    return articles

# Function to display previously generated articles in the sidebar
def display_articles_sidebar(articles):
    st.sidebar.header('Previously Generated Articles')
    for article in articles:
        lines = article.split('\n')
        topic = lines[0] if lines else "Unknown Topic"
        content = '\n'.join(lines[1:])
        with st.sidebar.expander(topic):
            st.text_area(label="", value=content, height=150)

# Streamlit app title
st.title('✍️Article Generator✍️')

articles = load_articles_from_file()
display_articles_sidebar(articles)

# Initialize the Anthropic client
client = ChatAnthropic(model="claude-3-haiku-20240307", api_key=os.environ.get("ANTHROPIC_API_KEY"), max_tokens=4096)

# User input for topic and article length
topic = st.text_input("Enter a topic for the article:")
num = st.text_input("Enter the Length (Words) of the articles:")

# Button to generate the initial article
if st.button("Generate Article on the given topic"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        try:
            with st.spinner("Generating initial article..."):
                prompt_template = "Write a detailed article about {topic} in {num} words."
                prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "num"])
                initial_article = client(prompt.format(topic=topic, num=num))
                #st.write("Initial Article:")
                #st.write(initial_article)

            with st.spinner("Fetching top articles..."):
                # Fetch top articles using Serper API
                search = GoogleSerperAPIWrapper(type='news')
                result_dict = search.results(f"Articles on {topic}")

                if not result_dict['news']:
                    st.error(f"No search results for: {topic}.")
                else:
                    # Initialize a variable to hold the article content
                    article_content = ""

                    # Fetch up to 10 articles
                    for i, item in zip(range(10), result_dict['news']):
                        loader = UnstructuredURLLoader(urls=[item['link']], ssl_verify=False)
                        data = loader.load()
                        if data:
                            document = data[0]
                            content = document.page_content
                            article_content += '\n\n\n\n' + content

            if article_content:
                with st.spinner("Analyzing and creating structure..."):
                    prompt_template = """Analyze the following articles separated by '/n/n/n/n' and create a best structure out of them for a comprehensive article:
                    {text}
                    """
                    cLient = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                    response = cLient.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=4096, temperature=0.5,
                        messages=[
                            {"role": "user", "content": prompt.format(text=article_content)}
                        ]
                    )
                    structure = response.content[0].text

            with st.spinner("Generating improved article..."):
                prompt_template = """You are an AI assistant tasked with improving an existing article based on a given structure and word count requirement. Your goal is to create a well-structured, informative, and engaging piece that expands upon the initial content while adhering to the provided guidelines.

                Here is the initial article you will be improving:

                <initial_article>
                {initial_article}
                </initial_article>

                The improved article should have a minimum word count of {num} words.

                Use the following structure as a guide for your improved article:

                <article_structure>
                {structure}
                </article_structure>

                To improve the article:

                1. Carefully read and analyze the initial article and the provided structure.
                2. Expand on each section of the structure, adding relevant information, examples, and details to meet the minimum word count.
                3. Ensure that the improved article covers all the points mentioned in the structure while maintaining a logical flow of ideas.
                4. Use the initial article as a foundation, but feel free to reorganize, rewrite, or add new content as needed to create a more comprehensive and engaging piece.
                5. Conduct additional research if necessary to provide accurate and up-to-date information on the topic.

                Follow these guidelines when writing the article:

                1. Adhere to the provided structure, using it as a framework for organizing your content.
                2. Ensure the article meets or exceeds the minimum word count specified.
                3. Give a human touch to the article by:
                   a. Using a conversational tone where appropriate
                   b. Including relatable examples or anecdotes
                   c. Addressing the reader directly when relevant
                   d. Using metaphors or analogies to explain complex concepts

                Once you have completed the outline, write the full article and enclose it in <article> and <headline> tags.

                Remember to proofread your work for grammar, spelling, and coherence before submitting your final version.
                Remember to stay focused on the main topic, expand on key points, and create a cohesive and informative article that meets the specified requirements."""

                prompt = PromptTemplate(template=prompt_template, input_variables=["initial_article", "structure", "num"])
                improved_article = client(prompt.format(initial_article=initial_article, structure=structure, num=num))
                st.write("Improved Article:")
                st.write(improved_article.content)
                save_article_to_file(topic,improved_article.content)

        except Exception as e:
            st.exception(f"Exception: {e}")
