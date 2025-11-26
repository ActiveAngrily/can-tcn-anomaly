import carla
import rospy
import json
import time
from std_msgs.msg import String

# --- CONFIGURATION ---
TOPIC_NAME = "/carla/ego_vehicle/can_data" 

def main():
    rospy.init_node('can_generator', anonymous=True)
    pub = rospy.Publisher(TOPIC_NAME, String, queue_size=10)
    print(f"[*] Searching for car to monitor...")

    try:
        # Connect to Docker Host
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
    except Exception as e:
        print(f"[FATAL] Could not connect to CARLA: {e}")
        return

    vehicle = None
    while not rospy.is_shutdown():
        actors = world.get_actors().filter('vehicle.*')
        if actors:
            vehicle = actors[0]
            print(f"✅ Vehicle Found! ID: {vehicle.id}")
            print(f"🚀 Broadcasting Normalized CAN data...")
            break
        time.sleep(1)

    rate = rospy.Rate(50) 
    
    # --- FIX: RELATIVE CLOCK ---
    start_time = time.time()
    
    while not rospy.is_shutdown():
        try:
            # Physics
            vel = vehicle.get_velocity()
            speed_kmh = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            control = vehicle.get_control()

            # Calculated Relative Timestamp (Starts at 0.0)
            current_timestamp = time.time() - start_time

            # Construct Packet
            can_packet = {
                "Timestamp": current_timestamp,    # FIXED: Now 0.0 -> 100.0+
                "CAN_ID": "0x18F02F01",
                "DLC": 8,
                "D0": int(speed_kmh),              
                "D1": int(control.throttle * 255), 
                "D2": int(control.brake * 255),    
                "D3": int((control.steer + 1) * 127), 
                "D4": int(vehicle.get_transform().rotation.yaw) % 255,
                "D5": 0, 
                "D6": 0, 
                "D7": 1 
            }

            json_msg = json.dumps(can_packet)
            pub.publish(json_msg)
            rate.sleep()
            
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == '__main__':
    main()