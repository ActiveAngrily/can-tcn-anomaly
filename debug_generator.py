import carla
import rospy
import json
import time
from std_msgs.msg import String

# --- CONFIGURATION ---
TOPIC_NAME = "/carla/ego_vehicle/can_data" 

def main():
    rospy.init_node('debug_generator', anonymous=True)
    pub = rospy.Publisher(TOPIC_NAME, String, queue_size=10)
    print(f"[*] Debug Generator Started. Searching for car...")

    try:
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
    except Exception as e:
        print(f"[FATAL] Connection failed: {e}")
        return

    vehicle = None
    while not rospy.is_shutdown():
        actors = world.get_actors().filter('vehicle.*')
        if actors:
            vehicle = actors[0]
            break
        time.sleep(1)

    print(f"✅ Found Vehicle. Starting Stream...")
    rate = rospy.Rate(2) # SLOW RATE (2 Hz) so you can read the logs
    
    # RESET CLOCK
    start_time = time.time()
    
    while not rospy.is_shutdown():
        try:
            vel = vehicle.get_velocity()
            speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            control = vehicle.get_control()
            
            # RELATIVE TIME CALCULATION
            now = time.time()
            rel_time = now - start_time

            packet = {
                "Timestamp": rel_time,    # <--- THIS SHOULD BE SMALL (e.g. 1.05)
                "CAN_ID": "0x18F02F01",   # Standard Hex ID
                "DLC": 8,
                "D0": int(speed),              
                "D1": int(control.throttle * 255), 
                "D2": int(control.brake * 255),    
                "D3": int((control.steer + 1) * 127), 
                "D4": int(vehicle.get_transform().rotation.yaw) % 255,
                "D5": 0, "D6": 0, "D7": 1 
            }

            # --- DEBUG PRINT ---
            print(f"\n[SENT] Time: {packet['Timestamp']:.4f} | Speed: {packet['D0']} | ID: {packet['CAN_ID']}")
            print(f"       Raw Packet: {json.dumps(packet)}")
            # -------------------

            pub.publish(json.dumps(packet))
            rate.sleep()
            
        except Exception as e:
            print(f"Loop Error: {e}")
            break

if __name__ == '__main__':
    main()