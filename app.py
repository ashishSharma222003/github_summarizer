import os
import streamlit as st
from constant import Gemini_api, github_token
from urllib.parse import urlparse
import requests
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import nest_asyncio
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SummaryIndex
from llama_index.core.chat_engine.types import ChatMode
import concurrent.futures

def load_documents(github_client, user, repo, excluded_dirs, branch):
    """Function to load documents from the GitHub repository."""
    return GithubRepositoryReader(
        github_client=github_client,
        owner=user,
        repo=repo,
        use_parser=True,
        verbose=False,
        filter_directories=(
            excluded_dirs,
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
        filter_file_extensions=(
            [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".ico",
                ".pdf",
                ".csv",
            ],
            GithubRepositoryReader.FilterType.EXCLUDE,
        )
    ).load_data(branch=branch)

def create_index(documents, transformations):
    """Function to create a vector store index from documents."""
    return VectorStoreIndex.from_documents(documents=documents, show_progress=True, transformations=transformations)

def create_summary(documents):
    """Function to create a summary index from documents."""
    return SummaryIndex.from_documents(documents=documents, show_progress=True)

def extract_github_details(url):
    """Extract the GitHub username and repository name from the URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    if path.count("/") == 1:
        user, repo = path.split("/", 1)
        return user, repo
    else:
        return None, None

def is_valid_github_url(url):
    parsed_url = urlparse(url)
    return (parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == "github.com" and
            parsed_url.path.startswith("/"))

def check_github_repo_exists(url):
    """Check if the GitHub repository exists."""
    try:
        path = urlparse(url).path.strip("/")
        user, repo = path.split("/", 1)
        api_url = f"https://github.com/{user}/{repo}"
        response = requests.get(api_url)
        return response.status_code == 200
    except ValueError:
        return False

def main():
    nest_asyncio.apply()
    st.markdown(
    """
    <div style="text-align: center;">
        <h1>Github Repo Summarizer and Helper</h1>
        <p>Created by <a href="https://github.com/ashishSharma222003" target="_blank">Ashish Sharma</a></p>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Description:</strong></p>
        <p>This project is designed to summarize and provide insights into GitHub repositories. You can input a GitHub repository link, and the application will fetch and display relevant information about the repository, including its README content, recent commits, and more. It aims to help users quickly understand the purpose and recent activity of GitHub repositories.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    st.divider()
    
    try:
        os.environ["GOOGLE_API_KEY"] = Gemini_api
    except ValueError as e:
        st.error("Google API key is not set.")
        return
    
    try:
        os.environ["GITHUB_TOKEN"] = github_token
    except ValueError as e:
        st.error("GitHub Token API key is not set.")
        return
    
    Settings.llm = Gemini()
    model_name = "models/embedding-001"
    Settings.embed_model = GeminiEmbedding(model_name=model_name, api_key=Gemini_api)
    
    # Initialize session state if not already present
    if 'github_client' not in st.session_state:
        st.session_state.github_client = None
        st.session_state.index = None
        st.session_state.summary = None
        st.session_state.query_engine = None
        st.session_state.summary_query_engine = None
        st.session_state.chat_history = []
    
    # Input for GitHub repo link
    github_repo_link = st.text_input(label="Enter GitHub Repo Link:", placeholder="https://github.com/UserName/repo-name")
    
    if github_repo_link:
        if not is_valid_github_url(github_repo_link):
            st.error("The URL format is incorrect. Please enter a valid GitHub repository URL.")
        else:
            if check_github_repo_exists(github_repo_link):
                if st.session_state.github_client is None:
                    st.session_state.github_client = GithubClient(github_token=github_token, verbose=True)
                
                user, repo = extract_github_details(github_repo_link)
                branch = st.text_input(label="Enter the branch you need to search:", placeholder="main")

                if branch:
                    st.markdown(
                    f"""
                    <div style="text-align: center; margin-bottom:10px">
                        <table style="margin: auto; border-collapse: collapse; width: 50%; border: 1px solid #ddd;">
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px;">User</th>
                                <td style="border: 1px solid #ddd; padding: 8px;">{user}</td>
                            </tr>
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px;">Repository</th>
                                <td style="border: 1px solid #ddd; padding: 8px;">{repo}</td>
                            </tr>
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px;">Branch</th>
                                <td style="border: 1px solid #ddd; padding: 8px;">{branch}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                
                exclude_dirs = st.checkbox("Exclude specific directories")
                excluded_dirs = []
                if exclude_dirs:
                    excluded_dirs_input = st.text_input(
                        label="Enter directories to exclude (comma-separated, optional):",
                        placeholder="dir1, dir2, dir3"
                    )
                    if excluded_dirs_input:
                        excluded_dirs = [dir.strip() for dir in excluded_dirs_input.split(',')]

                transformations = [
                    TokenTextSplitter(chunk_size=512, chunk_overlap=128),
                ]
                
                # Process data if not already processed
                if st.session_state.index is None or st.session_state.summary is None:
                    st.write("Please wait while the data is being processed...")
                    with st.spinner("Loading documents, creating index, and generating summary...."):
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future_documents = executor.submit(load_documents, st.session_state.github_client, user, repo, excluded_dirs, branch)
                            documents = future_documents.result()
                            future_index = executor.submit(create_index, documents, transformations)
                            future_summary = executor.submit(create_summary, documents)
                            st.session_state.index = future_index.result()
                            st.session_state.summary = future_summary.result()
                
                if st.session_state.query_engine is None:
                    st.session_state.query_engine = st.session_state.index.as_chat_engine(chat_mode=ChatMode.CONTEXT)
                if st.session_state.summary_query_engine is None:
                    st.session_state.summary_query_engine = st.session_state.summary.as_chat_engine(chat_mode=ChatMode.CONTEXT)
                
                st.markdown(
                    """
                    <style>
                    .full-width {
                        display: flex;
                        flex-direction: column;
                        width: 100%;
                    }
                    .full-width > div {
                        flex: 1;
                        padding: 10px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Display chat history
                for entry in st.session_state.chat_history:
                    with st.container():
                        st.write(f"**User:** {entry['question']}")
                        st.markdown('<div class="full-width">', unsafe_allow_html=True)
                        # st.write("Summary Response:", unsafe_allow_html=True)
                        # st.write(entry['summary_response'], unsafe_allow_html=True)
                        st.write("Chat Response:", unsafe_allow_html=True)
                        st.write(entry['achat_response'], unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Input for the user's question
                question = st.text_area(label="Write Your Question Here About The Code", key="question_input")

                if st.button("Submit Question"):
                    if question:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future_summary = executor.submit(st.session_state.summary_query_engine.query, question)
                            future_achat = executor.submit(st.session_state.query_engine.chat, question)

                            summary_response = future_summary.result()
                            achat_response = future_achat.result()

                        st.session_state.chat_history.append({
                            'question': question,
                            # 'summary_response': str(summary_response.response[0]),
                            'achat_response': str(achat_response.response)
                        })
                        # st.session_state.question_input = ""
                
            else:
                st.error("The GitHub repository does not exist or the URL is incorrect.")

if __name__ == "__main__":
    main()
