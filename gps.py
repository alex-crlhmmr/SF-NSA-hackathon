import serial

# Setup
SERIAL_PORT = '/dev/ttyACM0'  # Change to 'COM3' or similar if on Windows
BAUDRATE = 9600

# DOP thresholds for reliability
HDOP_THRESHOLD = 4.0  # Horizontal accuracy (speed, course)
VDOP_THRESHOLD = 4.0  # Vertical accuracy (altitude)

def parse_GGA(parts):
    """Parse $GPGGA sentence for position and time data."""
    try:
        if len(parts) < 15 or parts[0] != '$GPGGA':
            print(f"GGA invalid: {parts}")
            return {}
        utc_time = parts[1]
        lat = parts[2]
        lat_dir = parts[3]
        lon = parts[4]
        lon_dir = parts[5]
        fix = parts[6]
        num_satellites = parts[7]
        altitude = parts[9]

        # Convert latitude
        lat_deg = int(float(lat) / 100)
        lat_min = float(lat) - lat_deg * 100
        latitude = lat_deg + lat_min / 60.0
        if lat_dir == 'S':
            latitude = -latitude

        # Convert longitude
        lon_deg = int(float(lon) / 100)
        lon_min = float(lon) - lon_deg * 100
        longitude = lon_deg + lon_min / 60.0
        if lon_dir == 'W':
            longitude = -longitude

        time_str = f"{utc_time[0:2]}:{utc_time[2:4]}:{utc_time[4:6]}" if utc_time else "--"
        fix_status = "Yes" if fix in ['1', '2'] else "No"

        print(f"GGA parsed: Lat={latitude:.6f}, Lon={longitude:.6f}, Time={time_str}")  # Debug
        return {
            'UTC Time': time_str,
            'Latitude': round(latitude, 6),
            'Longitude': round(longitude, 6),
            'Fix': fix_status,
            'Satellites Used': num_satellites,
            'Altitude (m)': altitude if altitude else "--"
        }
    except Exception as e:
        print(f"GGA parsing error: {e}, parts: {parts}")
        return {}

def parse_RMC(parts):
    """Parse $GPRMC sentence for speed, course, and date."""
    try:
        if len(parts) < 10 or parts[0] != '$GPRMC':
            print(f"RMC invalid: {parts}")
            return {}
        status = "Valid" if parts[2] == 'A' else "Invalid"
        speed_knots = float(parts[7]) if parts[7] else 0.0
        course = float(parts[8]) if parts[8] else 0.0
        date = parts[9]
        date_formatted = f"{date[0:2]}/{date[2:4]}/20{date[4:6]}" if date else "--"
        print(f"RMC parsed: Speed={speed_knots} knots, Course={course} deg, Date={date_formatted}, Status={status}")  # Debug
        return {
            'Status': status,
            'Speed (km/h)': round(speed_knots * 1.852, 2),  # Convert knots to km/h
            'Course (deg)': round(course, 1),
            'Date': date_formatted
        }
    except Exception as e:
        print(f"RMC parsing error: {e}, parts: {parts}")
        return {}

def parse_VTG(parts):
    """Parse $GPVTG sentence for speed and course."""
    try:
        if len(parts) < 8 or parts[0] != '$GPVTG':
            print(f"VTG invalid: {parts}")
            return {}
        course = float(parts[1]) if parts[1] else 0.0
        speed_kph = float(parts[7]) if parts[7] else 0.0
        print(f"VTG parsed: Speed={speed_kph} km/h, Course={course} deg")  # Debug
        return {
            'Course (deg)': round(course, 1),
            'Speed (km/h)': round(speed_kph, 2)
        }
    except Exception as e:
        print(f"VTG parsing error: {e}, parts: {parts}")
        return {}

def parse_GSA(parts):
    """Parse $GPGSA sentence for fix type and DOP values."""
    try:
        if len(parts) < 18 or parts[0] != '$GPGSA':
            print(f"GSA invalid: {parts}")
            return {}
        fix_type = {'1': "No Fix", '2': "2D", '3': "3D"}.get(parts[2], "Unknown")
        pdop = parts[15] if parts[15] else "--"
        hdop = parts[16] if parts[16] else "--"
        vdop = parts[17] if parts[17] else "--"
        # Numeric values for correction logic with robust error handling
        try:
            pdop_num = float(parts[15]) if parts[15] else float('inf')
        except (ValueError, TypeError):
            pdop_num = float('inf')
        try:
            hdop_num = float(parts[16]) if parts[16] else float('inf')
        except (ValueError, TypeError):
            hdop_num = float('inf')
        try:
            vdop_num = float(parts[17]) if parts[17] else float('inf')
        except (ValueError, TypeError):
            vdop_num = float('inf')
        print(f"GSA parsed: Fix Type={fix_type}, PDOP={pdop}, HDOP={hdop}, VDOP={vdop}, Numeric={pdop_num}/{hdop_num}/{vdop_num}")  # Debug
        return {
            'Fix Type': fix_type,
            'PDOP': pdop,
            'HDOP': hdop,
            'VDOP': vdop,
            'PDOP_num': pdop_num,
            'HDOP_num': hdop_num,
            'VDOP_num': vdop_num
        }
    except Exception as e:
        print(f"GSA parsing error: {e}, parts: {parts}")
        return {}

def assess_corrected_values(gps_data):
    """Assess and return corrected values based on DOP thresholds."""
    hdop = gps_data.get('HDOP_num', float('inf'))
    vdop = gps_data.get('VDOP_num', float('inf'))
    fix_status = gps_data.get('Fix', 'No')
    satellites = int(gps_data.get('Satellites Used', 0))

    # Check reliability
    is_reliable = (
        hdop <= HDOP_THRESHOLD and
        vdop <= VDOP_THRESHOLD and
        fix_status == 'Yes' and
        satellites >= 4
    )

    # Corrected values
    corrected = {
        'Altitude (m)': gps_data.get('Altitude (m)', '--') if is_reliable and vdop <= VDOP_THRESHOLD else '--',
        'Speed (km/h)': gps_data.get('Speed (km/h)', '--') if is_reliable and hdop <= HDOP_THRESHOLD else '--',
        'Course (deg)': gps_data.get('Course (deg)', '--') if is_reliable and hdop <= HDOP_THRESHOLD else '--'
    }

    return corrected

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUDRATE} baud.\n")
    except Exception as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        return

    gps_data = {}

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Received: {line}")  # Debug
            parts = line.split(',')

            if line.startswith('$GPGGA'):
                gps_data.update(parse_GGA(parts))
            elif line.startswith('$GPRMC'):
                gps_data.update(parse_RMC(parts))
            elif line.startswith('$GPVTG'):
                gps_data.update(parse_VTG(parts))
            elif line.startswith('$GPGSA'):
                gps_data.update(parse_GSA(parts))

            if 'Latitude' in gps_data:
                # Get corrected values based on DOP
                corrected = assess_corrected_values(gps_data)

                print("\033c", end="")  # Clear terminal
                print("🌍  GPS Live Dashboard")
                print("=" * 40)
                print(f"🕓  UTC Time       : {gps_data.get('UTC Time', '--')}")
                print(f"📍  Latitude       : {gps_data.get('Latitude', '--')}")
                print(f"📍  Longitude      : {gps_data.get('Longitude', '--')}")
                print(f"📶  Fix Status     : {gps_data.get('Fix', '--')} ({gps_data.get('Fix Type', '--')})")
                print(f"🛰️  Satellites Used: {gps_data.get('Satellites Used', '--')}")
                print(f"⬆️  Altitude (m)   : {gps_data.get('Altitude (m)', '--')}")
                print(f"🎯  Speed (km/h)   : {gps_data.get('Speed (km/h)', '--')}")
                print(f"🧭  Course (deg)   : {gps_data.get('Course (deg)', '--')}")
                print(f"📅  Date           : {gps_data.get('Date', '--')}")
                print(f"📐  PDOP/HDOP/VDOP : {gps_data.get('PDOP', '--')} / {gps_data.get('HDOP', '--')} / {gps_data.get('VDOP', '--')}")
                print(f"🔧  Corrected Altitude (m) : {corrected['Altitude (m)']}")
                print(f"🔧  Corrected Speed (km/h) : {corrected['Speed (km/h)']}")
                print(f"🔧  Corrected Course (deg) : {corrected['Course (deg)']}")
                print("=" * 40)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()