import subprocess
import time
import os
import httpx
import asyncio

async def wait_for_proxy(url, timeout=120):
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                # Try root or analytics
                response = await client.get(f"{url}/v1/analytics", headers={"X-API-Key": "leo_admin_secret_key"})
                if response.status_code in [200, 404, 405]:
                    print(f"Proxy is up! (Status {response.status_code})")
                    return True
                else:
                    print(f"Proxy returned status {response.status_code}")
            except Exception as e:
                print(f"Waiting for proxy... ({e})")
            await asyncio.sleep(5)
    return False

async def main():
    # Set environment variables
    env = os.environ.copy()
    env["LEO_API_KEY"] = "leo_admin_secret_key"
    # OPENAI_API_KEY is already in env from sandbox
    
    print("Starting LEO Optima Proxy...")
    log_file = open("proxy_server.log", "w")
    proxy_process = subprocess.Popen(
        ["python3", "proxy_server.py"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    try:
        if await wait_for_proxy("http://localhost:8000"):
            print("Running stress tests...")
            test_process = subprocess.run(
                ["python3", "tests/new_stress_tests/comprehensive_stress_test.py"],
                env=env
            )
            if test_process.returncode == 0:
                print("Tests completed successfully!")
            else:
                print(f"Tests failed with return code {test_process.returncode}")
        else:
            print("Proxy failed to start within timeout.")
            stdout, stderr = proxy_process.communicate(timeout=5)
            print(f"Proxy stdout: {stdout}")
            print(f"Proxy stderr: {stderr}")
            
    finally:
        print("Shutting down proxy...")
        proxy_process.terminate()
        try:
            proxy_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_process.kill()

if __name__ == "__main__":
    asyncio.run(main())
