import carla
import time
import random

def inject_anomaly():
    print("------------------------------------------------")
    print("😈 ANOMALY INJECTOR: CONNECTING TO CARLA...")
    
    try:
        # 1. Connect to the Simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        # 2. Find the "Hero" Car (The one being driven)
        # We look for all vehicles and pick the main one (usually the first one)
        actors = world.get_actors().filter('vehicle.*')
        if not actors:
            print("❌ ERROR: No cars found! Is the simulation running?")
            return
            
        ego_vehicle = actors[0] # Assuming the first car found is ours
        print(f"✅ Target Acquired: {ego_vehicle.type_id}")
        
        # 3. Wait for dramatic effect
        print("⏳ Waiting 3 seconds before attack...")
        time.sleep(3)
        
        # 4. EXECUTE ATTACK (The "Swerve")
        print("🚀 INJECTING ANOMALY NOW! (Forcing Steer + Throttle)")
        
        # We override the controls: Full Throttle, Full Right Turn
        # This creates a data spike that the AI *definitely* didn't predict.
        control = carla.VehicleControl()
        control.throttle = 1.0
        control.steer = 1.0  # 1.0 is full right, -1.0 is full left
        control.brake = 0.0
        control.hand_brake = False
        
        # Apply this control for 1 second (enough to trigger the alarm)
        ego_vehicle.apply_control(control)
        time.sleep(1.0) 
        
        print("💀 Attack Complete. Returning control.")
        
        # Optional: Stop the car after the attack so it doesn't crash forever
        ego_vehicle.apply_control(carla.VehicleControl(brake=1.0))

    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        print("Make sure CARLA is running on this computer.")

if __name__ == '__main__':
    inject_anomaly()