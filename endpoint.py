import sys
import requests
import json
import base64
import time
import os
import asyncio
import websockets
from datetime import datetime
from typing import List, Dict, Any
from autogen import ConversableAgent

# Print websockets version for debugging
print(f"Using websockets version: {websockets.__version__}")

# Configuration
OUTPOST_IPS = {
    "outpost1": "http://10.1.61.197:8000",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Utility function to save base64 images
def save_base64_image(base64_str: str, outpost: str, prefix: str = "image") -> str:
    """Decode base64 string and save image as JPG."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{outpost}_{prefix}_{timestamp}.jpg"
        filepath = os.path.join(SCRIPT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(base64_str))
        return f"✅ Saved image: {filepath}"
    except Exception as e:
        return f"❌ Error saving image: {str(e)}"

# Core Functions
def validate_outposts(outposts: List[str]) -> List[str]:
    """Validate that all outposts are known."""
    valid_outposts = []
    for outpost in outposts:
        if outpost in OUTPOST_IPS:
            valid_outposts.append(outpost)
        else:
            print(f"Warning: Unknown outpost '{outpost}'. Skipping.")
    return valid_outposts if valid_outposts else list(OUTPOST_IPS.keys())

def take_picture(outposts: List[str]) -> Dict[str, Any]:
    outposts = validate_outposts(outposts)
    results = []
    images = []
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/take_picture"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            img_b64 = data.get("image")
            if img_b64:
                save_result = save_base64_image(img_b64, outpost, "picture")
                results.append(save_result)
                images.append(img_b64)
            else:
                results.append(f"⚠️ No image data from {outpost}")
        except Exception as e:
            results.append(f"❌ Error taking picture from {outpost}: {str(e)}")
    return {"message": "\n".join(results), "images": images}

def take_n_pictures(outposts: List[str], count: int = 10, interval: float = 2.0, annotated: bool = True, quality: int = 15) -> Dict[str, Any]:
    outposts = validate_outposts(outposts)
    results = []
    all_images = []
    params = {"count": count, "interval": interval, "annotated": annotated, "quality": quality}
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/take_n_pictures"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            pictures = data.get("pictures", [])
            saves = [save_base64_image(img, outpost, f"n_pictures_{i+1}") for i, img in enumerate(pictures) if img]
            results.append(f"📸 {len(pictures)} pictures from {outpost}: {', '.join(saves)}")
            all_images.extend(pictures)
        except Exception as e:
            results.append(f"❌ Error taking multiple pictures from {outpost}: {str(e)}")
    return {"message": "\n".join(results), "images": all_images}

def take_n_pictures_with_info(outposts: List[str], count: int = 10, interval: float = 2.0, include_image: bool = True, annotated: bool = True, quality: int = 15) -> Dict[str, Any]:
    outposts = validate_outposts(outposts)
    results = []
    all_images = []
    all_detections = []
    params = {"count": count, "interval": interval, "include_image": include_image, "annotated": annotated, "quality": quality}
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/take_n_pictures_with_info"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            captures = data.get("captures", [])
            saves = []
            for i, capture in enumerate(captures):
                img_b64 = capture.get("image")
                detected_data = capture.get("detected_data", {})
                save_result = save_base64_image(img_b64, outpost, f"n_pictures_info_{i+1}") if img_b64 and include_image else "No image included"
                saves.append(f"Capture {i+1}: {save_result}, Detection: {json.dumps(detected_data, indent=2)}")
                if img_b64 and include_image:
                    all_images.append(img_b64)
                    all_detections.append(detected_data)
            results.append(f"📸 {len(captures)} pictures with info from {outpost}:\n" + "\n".join(saves))
        except Exception as e:
            results.append(f"❌ Error taking pictures with info from {outpost}: {str(e)}")
    return {"message": "\n".join(results), "images": all_images, "detections": all_detections}

def take_picture_with_info(outposts: List[str], include_image: bool = True) -> Dict[str, Any]:
    outposts = validate_outposts(outposts)
    results = []
    images = []
    detections = []
    params = {"include_image": include_image}
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/take_picture_with_info"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            img_b64 = data.get("image")
            detected_data = data.get("detected_data", {})
            if img_b64 and include_image:
                save_result = save_base64_image(img_b64, outpost, "picture_info")
                results.append(f"✅ Picture with info from {outpost}: {save_result}\nDetection: {json.dumps(detected_data, indent=2)}")
                images.append(img_b64)
                detections.append(detected_data)
            else:
                results.append(f"⚠️ Picture with info from {outpost}: No image data\nDetection: {json.dumps(detected_data, indent=2)}")
        except Exception as e:
            results.append(f"❌ Error taking picture with info from {outpost}: {str(e)}")
    return {"message": "\n".join(results), "images": images, "detections": detections}

def latest_detection(outposts: List[str]) -> str:
    outposts = validate_outposts(outposts)
    results = []
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/latest_detection"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            detection_data = response.json().get("latest_detection", {})
            results.append(f"🕵️ Latest detection for {outpost}: {json.dumps(detection_data, indent=2)}")
        except Exception as e:
            results.append(f"❌ Error getting latest detection from {outpost}: {str(e)}")
    return "\n".join(results)

def gps_info(outposts: List[str]) -> str:
    outposts = validate_outposts(outposts)
    results = []
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/gps_info"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            gps_data = data.get("gps_data", {})
            corrected_values = data.get("corrected_values", {})
            results.append(f"📍 GPS info for {outpost}:\nGPS Data: {json.dumps(gps_data, indent=2)}\nCorrected Values: {json.dumps(corrected_values, indent=2)}")
        except Exception as e:
            results.append(f"❌ Error getting GPS info from {outpost}: {str(e)}")
    return "\n".join(results)

def reset_camera(outposts: List[str]) -> str:
    outposts = validate_outposts(outposts)
    results = []
    for outpost in outposts:
        url = f"{OUTPOST_IPS[outpost]}/reset_camera"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            status = response.json().get("status", "Unknown")
            results.append(f"🔄 Camera reset for {outpost}: {status}")
        except Exception as e:
            results.append(f"❌ Error resetting camera at {outpost}: {str(e)}")
    return "\n".join(results)

def video_feed(outposts: List[str]) -> str:
    outposts = validate_outposts(outposts)
    results = []
    for outpost in outposts:
        feed_url = f"{OUTPOST_IPS[outpost]}/video_feed"
        results.append(f"🎥 Video feed for {outpost}: {feed_url}")
    return "\n".join(results)

def analyze_outpost(outposts: List[str]) -> str:
    """Analyze the latest image and detection data from the outpost using Qwen's multimodal capabilities."""
    outposts = validate_outposts(outposts)
    results = []
    for outpost in outposts:
        try:
            # Fetch the latest image and detection data
            url = f"{OUTPOST_IPS[outpost]}/take_picture_with_info"
            params = {"include_image": True}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            img_b64 = data.get("image")
            detected_data = data.get("detected_data", {})

            if not img_b64:
                results.append(f"⚠️ No image data available for {outpost}")
                continue

            # Prepare input for Qwen using Ollama's images field
            prompt = (
                f"Analyze the current situation at {outpost}. Describe what is happening based on the provided image and the following detection data: {json.dumps(detected_data, indent=2)}. "
                "Provide a detailed description of the scene, any objects or activities detected, and any relevant context from the detection data."
            )
            message = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Use the images field for multimodal input
            try:
                reply = assistant.generate_reply(messages=message, images=[img_b64])
                if isinstance(reply, dict) and "content" in reply:
                    results.append(f"🔍 Analysis for {outpost}:\n{reply['content']}")
                elif isinstance(reply, str):
                    results.append(f"🔍 Analysis for {outpost}:\n{reply}")
                else:
                    results.append(f"❌ Error analyzing {outpost}: Invalid response format")
            except Exception as e:
                results.append(f"❌ Error analyzing {outpost}: Model may not support multimodal input - {str(e)}")
        except Exception as e:
            results.append(f"❌ Error fetching data for {outpost}: {str(e)}")
    return "\n".join(results)

def execute_sequence(actions: List[Dict[str, Any]]) -> str:
    results = []
    if not isinstance(actions, list):
        return f"❌ Error: Actions must be a list, got {type(actions)}"

    FUNCTION_MAP = {
        "take_picture": take_picture,
        "take_n_pictures": take_n_pictures,
        "take_n_pictures_with_info": take_n_pictures_with_info,
        "take_picture_with_info": take_picture_with_info,
        "latest_detection": latest_detection,
        "gps_info": gps_info,
        "reset_camera": reset_camera,
        "video_feed": video_feed,
        "analyze_outpost": analyze_outpost,
    }

    for step in actions:
        if not isinstance(step, dict):
            results.append(f"❌ Error: Action must be a dictionary, got {type(step)}")
            continue
        action = step.get("action") or step.get("function")
        args = step.get("args", {})
        if action not in FUNCTION_MAP:
            results.append(f"❌ Unknown action: {action}")
            continue
        try:
            if "outposts" not in args:
                args["outposts"] = list(OUTPOST_IPS.keys())
            result = FUNCTION_MAP[action](**args)
            if action in ["take_picture", "take_n_pictures", "take_n_pictures_with_info", "take_picture_with_info"]:
                results.append(f"✅ {action}: {result['message']}")
            else:
                results.append(f"✅ {action}: {result}")
        except Exception as e:
            results.append(f"❌ Error executing {action}: {str(e)}")
    return "\n".join(results)

# Ollama Configuration
config_list = [
    {
        "model": "qwen2.5:3b",
        "api_type": "ollama",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",
    }
]

llm_config = {"config_list": config_list}

# Initialize the ConversableAgent
assistant = ConversableAgent(
    name="CameraGPSOperator",
    system_message=(
        "You are a camera and GPS control agent managing multiple outposts: " + ", ".join(OUTPOST_IPS.keys()) + ". "
        "Available functions: take_picture, take_n_pictures, take_n_pictures_with_info, take_picture_with_info, "
        "latest_detection, gps_info, reset_camera, video_feed, analyze_outpost, execute_sequence. "
        "Each function requires an 'outposts' parameter (e.g., ['outpost1']). If no outposts are specified, use all outposts: " + str(list(OUTPOST_IPS.keys())) + ". "
        "For 'get gps outpost1', call gps_info with outposts=['outpost1']. "
        "For 'video feed', return the video feed URL for display. "
        "For 'take picture' or 'take n pictures', return image data for display. "
        "For 'what's going on outpost1', call analyze_outpost to analyze the latest image and detection data. "
        "For multiple actions (e.g., 'get gps and take a picture'), use execute_sequence. "
        "Only include parameters explicitly mentioned in the user's request. Use defaults for unspecified parameters. "
        "For clarification or info requests, respond politely without tool calls."
    ),
    llm_config=llm_config,
    human_input_mode="AUTO",
    functions=[
        take_picture,
        take_n_pictures,
        take_n_pictures_with_info,
        take_picture_with_info,
        latest_detection,
        gps_info,
        reset_camera,
        video_feed,
        analyze_outpost,
        execute_sequence,
    ],
    function_map={
        "take_picture": take_picture,
        "take_n_pictures": take_n_pictures,
        "take_n_pictures_with_info": take_n_pictures_with_info,
        "take_picture_with_info": take_picture_with_info,
        "latest_detection": latest_detection,
        "gps_info": gps_info,
        "reset_camera": reset_camera,
        "video_feed": video_feed,
        "analyze_outpost": analyze_outpost,
        "execute_sequence": execute_sequence,
    },
)

# WebSocket Server
async def handle_websocket(websocket, path: str = "/"):
    print(f"New WebSocket connection: {path}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            try:
                data = json.loads(message)
                user_input = data.get("message", "")
                if user_input.lower() in ("exit", "quit"):
                    await websocket.send(json.dumps({"type": "response", "message": "Session ended"}))
                    break

                reply = assistant.generate_reply(messages=[{"role": "user", "content": user_input}])
                response_data = {"type": "response", "message": ""}

                if isinstance(reply, dict):
                    if "content" in reply and reply["content"]:
                        response_data["message"] = reply["content"]
                    elif "tool_calls" in reply and reply["tool_calls"]:
                        for tool_call in reply["tool_calls"]:
                            func_name = tool_call["function"]["name"]
                            args_str = tool_call["function"]["arguments"]
                            print(f"Tool call: {func_name}, Args: {args_str}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError as e:
                                response_data["message"] = f"Error parsing arguments for {func_name}: {str(e)}"
                                break
                            if func_name in assistant.function_map:
                                result = assistant.function_map[func_name](**args)
                                if func_name == "take_picture":
                                    response_data["type"] = "image"
                                    response_data["message"] = result["message"]
                                    response_data["images"] = result["images"]
                                elif func_name == "take_n_pictures":
                                    response_data["type"] = "images"
                                    response_data["message"] = result["message"]
                                    response_data["images"] = result["images"]
                                elif func_name == "take_n_pictures_with_info":
                                    response_data["type"] = "images"
                                    response_data["message"] = result["message"]
                                    response_data["images"] = result["images"]
                                    response_data["detections"] = result["detections"]
                                elif func_name == "take_picture_with_info":
                                    response_data["type"] = "image"
                                    response_data["message"] = result["message"]
                                    response_data["images"] = result["images"]
                                    response_data["detections"] = result["detections"]
                                elif func_name == "video_feed":
                                    response_data["type"] = "video_feed"
                                    response_data["message"] = result
                                    response_data["url"] = result.split(": ")[-1] if ": " in result else result
                                else:
                                    response_data["message"] = result
                            else:
                                response_data["message"] = f"Error - Unknown function {func_name}"
                    else:
                        response_data["message"] = "Error - No valid response content or tool calls"
                elif isinstance(reply, str):
                    response_data["message"] = reply
                else:
                    response_data["message"] = f"Error - Invalid response format: {type(reply)}"

                print(f"Sending response: {response_data}")
                await websocket.send(json.dumps(response_data))
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                await websocket.send(json.dumps({"type": "response", "message": f"Error: {str(e)}"}))
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {str(e)}")

# Start WebSocket Server
async def main():
    try:
        websocket_server = await websockets.serve(handle_websocket, "localhost", 8765)
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # Keep the server running
    except Exception as e:
        print(f"Error starting WebSocket server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())