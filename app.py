
import os
import re
import streamlit as st

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="YouTube to Blog AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube to Blog Generator")
st.write(
    "Convert YouTube videos into professional blog articles "
    "using CrewAI and Groq AI."
)


# --------------------------------
# 1. Extract YouTube Video ID
# --------------------------------

def get_video_id(url):

    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# --------------------------------
# 2. Extract YouTube Transcript
# --------------------------------

def get_transcript(video_id):

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # Try English first, then Telugu
    try:
        transcript = transcript_list.find_transcript(["en"])

    except Exception:
        try:
            transcript = transcript_list.find_transcript(["te"])

        except Exception:
            # Try other available generated transcripts
            transcript = transcript_list.find_generated_transcript(
                ["en", "te"]
            )

    fetched_transcript = transcript.fetch()

    transcript_text = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    return transcript_text


# --------------------------------
# 3. Generate Blog Using CrewAI
# --------------------------------

def generate_blog(transcript, topic):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    # Configure Groq LLM
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=api_key
    )

    # Research Agent
    researcher = Agent(
        role="Content Research Analyst",

        goal=(
            "Analyze YouTube transcripts and identify "
            "important information, key ideas and examples."
        ),

        backstory=(
            "You are an expert content researcher. "
            "You extract useful information from video transcripts."
        ),

        llm=llm,
        verbose=False
    )

    # Blog Writer Agent
    writer = Agent(
        role="Professional Blog Writer",

        goal=(
            "Create engaging, informative and well-structured "
            "blog articles from research findings."
        ),

        backstory=(
            "You are an experienced professional blog writer "
            "who creates high-quality articles."
        ),

        llm=llm,
        verbose=False
    )

    # Research Task
    research_task = Task(

        description=(
            "Analyze the following YouTube transcript about {topic}.\n\n"

            "Identify:\n"
            "1. Main ideas\n"
            "2. Important points\n"
            "3. Key facts\n"
            "4. Examples\n"
            "5. Important takeaways\n\n"

            "Use only information supported by the transcript.\n\n"

            "TRANSCRIPT:\n{transcript}"
        ),

        expected_output=(
            "A detailed research report containing "
            "main ideas, facts, examples and key takeaways."
        ),

        agent=researcher
    )

    # Blog Writing Task
    writing_task = Task(

        description=(
            "Write a complete professional blog article about {topic} "
            "using the research report from the previous task.\n\n"

            "Include:\n"
            "1. Attractive blog title\n"
            "2. Introduction\n"
            "3. Main headings and subheadings\n"
            "4. Detailed explanations\n"
            "5. Conclusion\n"
            "6. Key takeaways\n\n"

            "Write in clear English. Do not invent facts."
        ),

        expected_output=(
            "A complete, well-formatted Markdown blog article."
        ),

        agent=writer,

        context=[research_task]
    )

    # Create Crew
    crew = Crew(

        agents=[
            researcher,
            writer
        ],

        tasks=[
            research_task,
            writing_task
        ],

        process=Process.sequential,

        verbose=False
    )

    # Execute Crew
    result = crew.kickoff(

        inputs={
            "topic": topic,
            "transcript": transcript
        }
    )

    return str(result)


# --------------------------------
# 4. Streamlit User Interface
# --------------------------------

video_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

topic = st.text_input(
    "Blog Topic (Optional)",
    placeholder="Example: Artificial Intelligence"
)


# Generate Blog Button
if st.button("🚀 Generate Blog", type="primary"):

    if not video_url.strip():

        st.error("Please enter a YouTube video URL.")

    else:

        video_id = get_video_id(video_url)

        if not video_id:

            st.error("Invalid YouTube URL.")

        else:

            try:

                # Extract Transcript
                with st.spinner("Extracting YouTube transcript..."):

                    transcript = get_transcript(video_id)

                if not transcript.strip():

                    st.error("Transcript is empty.")

                else:

                    st.success(
                        "YouTube transcript extracted successfully!"
                    )

                    with st.expander("View YouTube Transcript"):

                        st.write(transcript)

                    # Generate Blog
                    with st.spinner(
                        "AI Agents are researching and writing your blog..."
                    ):

                        blog = generate_blog(
                            transcript,
                            topic.strip() or "Video Summary"
                        )

                    st.success("Blog generated successfully!")

                    st.subheader("📝 Generated Blog")

                    st.markdown(blog)

                    # Download Blog
                    st.download_button(

                        label="📥 Download Blog",

                        data=blog,

                        file_name="youtube_blog.md",

                        mime="text/markdown"
                    )

            except Exception as error:

                st.error(f"Error: {error}")

                st.info(
                    "Check your YouTube captions, Groq API key "
                    "and model availability."
                )
