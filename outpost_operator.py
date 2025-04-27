import sys
import json
from autogen import ConversableAgent

# --- Ollama config ---
config_list = [
    {
        "model": "llama3.1:8b-instruct-fp16",
        "api_type": "ollama",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    }
]
llm_config = {"config_list": config_list}

# --- Map outposts and functions ---
OUTPOST_IPS = {"outpost1": "http://10.1.61.197:8000"}

# Import your function implementations here
from functions import (
    take_picture,
    take_n_pictures,
    take_n_pictures_with_info,
    take_picture_with_info,
    latest_detection,
    gps_info,
    reset_camera,
    video_feed,
    execute_sequence,
)

FUNCTION_MAP = {
    "take_picture": take_picture,  
    "take_n_pictures": take_n_pictures,  
    "take_n_pictures_with_info": take_n_pictures_with_info,  
    "take_picture_with_info": take_picture_with_info,  
    "latest_detection": latest_detection,  
    "gps_info": gps_info,  
    "reset_camera": reset_camera,  
    "video_feed": video_feed,  
    "execute_sequence": execute_sequence,  
}

# --- System Message: Teach the LLM how to interpret user intent ---
system_message = (
    "You are an outpost control agent. The user will speak in natural language. "
    "Your role is to understand their intent and trigger the correct function with the right arguments. "
    "DO NOT expect exact keywords. Adapt to human phrasing like 'snap a photo,' 'show me the GPS,' "
    "or 'take some annotated pictures.' If unsure, ASK for clarification. "
    "Functions available:\n"
    "- take_picture(outposts: List[str]) -> Take one picture.\n"
    "- take_n_pictures(outposts: List[str], count: int, interval: float, annotated: bool, quality: int) -> Take multiple pictures.\n"
    "- take_n_pictures_with_info(outposts: List[str], count: int, interval: float, include_image: bool, annotated: bool, quality: int) -> Take multiple pictures with detection info.\n"
    "- take_picture_with_info(outposts: List[str], include_image: bool) -> Take a picture with detection info.\n"
    "- latest_detection(outposts: List[str]) -> Get the latest detection data.\n"
    "- gps_info(outposts: List[str]) -> Get GPS position.\n"
    "- reset_camera(outposts: List[str]) -> Reset the camera.\n"
    "- video_feed(outposts: List[str]) -> Get video feed URL.\n"
    "- execute_sequence(actions: List[Dict[str, Any]]) -> Run a sequence of actions.\n"
    "If the input is unclear, ASK the user for clarification instead of guessing."
)

# --- Initialize the ConversableAgent ---
assistant = ConversableAgent(
    name="OutpostOperator",
    system_message=system_message,
    llm_config=llm_config,
    human_input_mode="NEVER", 
    function_map=FUNCTION_MAP,
    code_execution_config=False
)

# --- Main Loop ---
if __name__ == "__main__":
    print("Type 'exit' or 'quit' to quit.")
    print("Try natural commands like: 'snap a few pics from outpost1 and show me the GPS.'")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            reply = assistant.generate_reply(messages=[{"role": "user", "content": user_input}])
            if isinstance(reply, dict) and "content" in reply and reply["content"]:
                print("Assistant:", reply["content"])
            elif isinstance(reply, str):
                print("Assistant:", reply)
            else:
                print("Assistant: Sorry, I didn't understand. Could you rephrase?")
        except Exception as e:
            print(f"Assistant: Error generating response: {e}")
