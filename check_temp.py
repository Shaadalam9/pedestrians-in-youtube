import subprocess
import os
import time
import threading
from logmod import logs
from custom_logger import CustomLogger

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class TemperatureMonitor(threading.Thread):
    def __init__(self, interval_minutes=5, temp_threshold=80):
        super().__init__()
        self.interval_minutes = interval_minutes
        self.temp_threshold = temp_threshold
        self.running = False  # Control flag to stop the thread

    def run(self):
        """Main loop of the thread."""
        self.running = True
        while self.running:
            self.check_temperature()
            time.sleep(self.interval_minutes * 60)

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False

    def check_temperature(self):
        """Check sensor data and shut down if the temperature exceeds the threshold."""
        try:
            # Run the `sensors` command and capture the output
            output = subprocess.check_output(["sensors"], encoding="utf-8")

            # Parse the output and check for temperatures exceeding the threshold
            for line in output.splitlines():
                if "°C" in line:
                    # Extract temperature value
                    temp_str = line.split()[1].strip("°C").strip("+")
                    try:
                        temp = float(temp_str)
                        if temp > self.temp_threshold:
                            print(f"High temperature detected: {temp}°C. Shutting down the system!")
                            os.system("shutdown now")
                    except ValueError:
                        continue

        except subprocess.CalledProcessError as e:
            print(f"Error while running sensors: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# Example Usage in Another Script
if __name__ == "__main__":
    monitor = TemperatureMonitor(interval_minutes=5, temp_threshold=80)
    monitor.start()  # Start monitoring in a separate thread

    try:
        while True:
            # Simulate main program logic here
            logger.info("Main program is running...")
            time.sleep(10)  # Keep the main program running
    except KeyboardInterrupt:
        logger.info("Stopping temperature monitor...")
        monitor.stop()  # Stop the thread
        monitor.join()  # Wait for the thread to finish
        logger.info("Monitor stopped.")
