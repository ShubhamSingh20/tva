from tva import GeminiLLM, VIDEO_ANALYSIS_PROMPT
from dotenv import load_dotenv

load_dotenv()

def test_gemini_vision():
    llm = GeminiLLM()
    chunk_path = "chunks/chunk_001_padded.mp4"

    prompt = f"""
    USER TASK: Locate and click on the Search icon on the homepage. Verify that the search bar is displayed and ready for input.
    """
    response = llm.generate_response(prompt, VIDEO_ANALYSIS_PROMPT, [chunk_path])
    print(response)

if __name__ == "__main__":
    test_gemini_vision()
