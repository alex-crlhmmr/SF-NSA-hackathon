import socket
import uuid
from datetime import datetime, timedelta, timezone
import time
from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP("civtak")

#IP 1: 161.35.231.249, Port: 8087 [Marton's remote machine]
#IP 2: 10.1.60.174, Port: 4242 [Grace's local machine]
FREE_TAK_SERVER_IP = "161.35.231.249"
FREE_TAK_SERVER_PORT = 8087

def parse_ais_record(ais_data: dict) -> dict:
    """
    Extracts critical fields from an AIS data dictionary.
    
    Args:
        ais_data (dict): Dictionary containing AIS record fields.
    
    Returns:
        dict: { 'lat': float, 'lon': float, 'mmsi': str, 'vessel_name': str }
    """
    try:
        lat = float(ais_data.get('LAT', 0.0))
        lon = float(ais_data.get('LON', 0.0))
        mmsi = ais_data.get('MMSI', 'Unknown')
        vessel_name = ais_data.get('VesselName', 'Unknown Vessel')
        
        return {
            'lat': lat,
            'lon': lon,
            'mmsi': mmsi,
            'vessel_name': vessel_name
        }
    except Exception as e:
        print(f"[!] Error parsing AIS record: {e}")
        return {}

def generate_random_uid(prefix="event"):
    return f"{prefix}-{uuid.uuid4().hex[:6]}"

def generate_fresh_timestamps():
    now = datetime.now(timezone.utc)
    start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return time_now, start, stale

def build_cot(uid: str, lat: float, lon: float, remarks: str):
    time_now, start, stale = generate_fresh_timestamps()
    
    cot = f'''
<event version="2.0" uid="{uid}" type="b-t-u" how="m-g" time="{time_now}" start="{start}" stale="{stale}">
  <point lat="{lat}" lon="{lon}" hae="0" ce="10.0" le="5.0"/>
  <detail>
    <contact callsign="{uid}"/>
    <remarks>{remarks}</remarks>
    <__group role="Team Member" name="CIV" />
  </detail>
</event>
'''
    return cot

def generate_tak_hello():
    now = datetime.now(timezone.utc)
    time_now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'''
<event version="2.0" uid="hello-client-{uuid.uuid4().hex[:4]}" type="t-x-c-t" how="m-g" time="{time_now}" start="{time_now}" stale="{stale}">
  <point lat="0.0" lon="0.0" hae="0" ce="9999999" le="9999999"/>
  <detail><contact callsign="AutoHello" /><takv device="FakeDevice" platform="Python" os="Linux" version="1.0"/></detail>
</event>
'''

async def send_cot_to_freetakserver(cot_xml: str, server_ip: str = FREE_TAK_SERVER_IP, server_port: int = FREE_TAK_SERVER_PORT):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((server_ip, server_port))
            
            hello_xml = generate_tak_hello()
            sock.sendall(hello_xml.encode('utf-8'))
            time.sleep(0.3)
            
            sock.sendall(cot_xml.encode('utf-8'))
            print(f"[+] CoT sent to FreeTAKServer at {server_ip}:{server_port}")
            
        return "Successfully sent CoT."
    except Exception as e:
        print(f"[!] Error sending CoT: {e}")
        return f"Failed to send CoT: {e}"

@mcp.tool()
async def send_image_marker(lat: float, lon: float, image_url: str) -> str:
    """Send a marker to CIVTAK with a lat/lon and linked image."""
    uid = generate_random_uid("img")
    remarks = f"Image link: {image_url}"
    cot = build_cot(uid, lat, lon, remarks)
    result = await send_cot_to_freetakserver(cot)
    return result

@mcp.tool()
async def send_ais_marker(lat: float, lon: float, vessel_name: str, mmsi: str) -> str:
    """Send a marker to CIVTAK for a ship detected via AIS."""
    uid = vessel_name
    remarks = f"AIS Contact - Ship: {vessel_name}, MMSI: {mmsi}"
    cot = build_cot(uid, lat, lon, remarks)
    result = await send_cot_to_freetakserver(cot)
    return result

@mcp.tool()
async def send_rf_detection_marker(lat: float, lon: float, frequency_mhz: float, remarks: str) -> str:
    """Send a marker to CIVTAK for an RF signal detection."""
    uid = generate_random_uid("rf")
    cot = build_cot(uid, lat, lon, remarks)
    result = await send_cot_to_freetakserver(cot)
    return result

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
