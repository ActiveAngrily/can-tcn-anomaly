import carla
import time
import random

def inject_anomaly():
    print("------------------------------------------------")
    print("😈 ANOMALY INJECTOR: CONNECTING TO CARLA...")
    
    try:
        # 1. Connect to the Simulator
        # Use 'host.docker.internal' for Docker, 'localhost' for local
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        # IMPORTANT: Sync with the server to ensure we find the car
        world.wait_for_tick()

        # 2. Find the "Hero" Car
        actors = world.get_actors().filter('vehicle.*')
        if not actors:
            print("❌ ERROR: No cars found! Is the simulation running?")
            return
            
        ego_vehicle = actors[0] 
        print(f"✅ Target Acquired: {ego_vehicle.type_id} (ID: {ego_vehicle.id})")
        
        # 3. Countdown
        print("⏳ Inserting CHAOS in 3 seconds...")
        time.sleep(3)
        
        # 4. EXECUTE "FISH-TAIL" ATTACK
        # We alternate steering rapidly to confuse the model
        print("🚀 INJECTING ANOMALY: VIOLENT INSTABILITY!")
        
        # Disable Autopilot
        ego_vehicle.set_autopilot(False)

        # Parameters
        duration = 2.0  # Attack lasts 2 seconds
        start_time = time.time()
        
        # Loop to overpower ROS bridge
        while time.time() - start_time < duration:
            # Oscillate Steering: Left -> Right -> Left
            # We use time to flip the steering every 0.2 seconds
            if int(time.time() * 5) % 2 == 0:
                steer_val = -0.8 # Hard Left
            else:
                steer_val = 0.8  # Hard Right
                
            # Apply Control
            # Throttle is high to maintain chaos
            control = carla.VehicleControl(throttle=0.8, steer=steer_val, brake=0.0)
            ego_vehicle.apply_control(control)
            
            # Sleep briefly to let physics react, but fast enough to beat ROS
            time.sleep(0.05)

        print("💀 Attack Complete. Stabilizing...")
        
        # 5. Recovery Phase (Coast to stop)
        for _ in range(20):
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.2))
            time.sleep(0.05)

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == '__main__':
    inject_anomaly()