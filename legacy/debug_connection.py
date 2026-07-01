import carla
import time

def test_connection():
    print("[*] Connecting to CARLA at host.docker.internal:2000...")
    try:
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(5.0)
        
        # 1. Get World and Map Name
        world = client.get_world()
        
        print(f"✅ Connection established!")
        print(f"   - Current Map: {world.get_map().name}")
        
        # 2. Force a Synchronization Tick
        # Sometimes the client needs to 'hear' from the server to update the actor list
        print("   - Waiting for server tick...")
        world.wait_for_tick()
        
        # 3. Count Actors
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        
        print(f"   - Total Actors: {len(actors)}")
        print(f"   - Vehicles:     {len(vehicles)}")
        
        if len(vehicles) > 0:
            print(f"   - Found Vehicle ID: {vehicles[0].id}")
            print("🎉 SUCCESS! The injector should work now.")
        else:
            print("❌ FAILURE: Still no vehicles found.")
            print("   POSSIBLE FIX: Use your Mac's actual IP address instead of 'host.docker.internal'.")

    except Exception as e:
        print(f"❌ CRASH: {e}")

if __name__ == '__main__':
    test_connection()