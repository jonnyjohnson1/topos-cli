import time
import uuid
import asyncio

# EXAMPLE GOOGLE CLOUD LOGGER
# import google.cloud.logging
# from google.cloud.logging_v2 import Client

# Initialize the Google Cloud Logging client
# logging_client = google.cloud.logging.Client()
# logger = logging_client.logger("process-logger")  # Name of the log

# TODO MAKE DEFAULT LOGGER USE POSTGRES

class ProcessLogger:
    def __init__(self, verbose=False, run_logger=False):
        self.logs = {}
        self.run_logger = run_logger
        self.verbose = verbose
        self.step_id = uuid.uuid4().hex

    async def submit_logs(self, log_message):
        """Internal method to submit logs to Google Cloud Logging asynchronously."""
        if self.verbose:
            print(log_message)
        # TODO CREATE LOGGER CLIENT AND SERVICE
        # if self.run_logger:
            # Run the log submission in a separate thread to avoid blocking
            # await asyncio.to_thread(logger.log_text, log_message)
    
    async def log(self, log_message):
        """Internal method to submit logs to Google Cloud Logging asynchronously."""
        # Run the log submission in a separate thread to avoid blocking
        # TODO CREATE LOGGER CLIENT AND SERVICE
        # await asyncio.to_thread(logger.log_text, log_message)

    async def start(self, step_name, **kwargs):
        start_time = time.time()
        self.logs[step_name] = {
            "start_time": start_time,
            "details": kwargs if kwargs else {}
        }
        
        # Format the details as a string delimited by |
        # details = '|'.join([f"{key}={value}" for key, value in kwargs.items()]) if kwargs else ""
        # # Create the log message
        # log_message = f"{step_name},{self.step_id},{start_time},,,{details}"
        
        # # Submit the log if run_logger is True
        # await self.submit_logs(log_message)

    async def end(self, step_name, **kwargs):
        if step_name in self.logs:
            end_time = time.time()
            elapsed_time = end_time - self.logs[step_name]["start_time"]
            self.logs[step_name]["end_time"] = end_time
            self.logs[step_name]["elapsed_time"] = elapsed_time
            
            # Update the log with any additional details passed in at the end
            if kwargs:
                if self.logs[step_name]["details"]:
                    self.logs[step_name]["details"].update(kwargs)
                else:
                    self.logs[step_name]["details"] = kwargs
            
            # Format the details as a string delimited by |
            details = '|'.join([f"{key}={value}" for key, value in self.logs[step_name].get("details", {}).items()])
            
            # Create the log message
            log_message = (
                f"{step_name},{self.step_id},"
                f"{self.logs[step_name]['start_time']},{end_time},{elapsed_time},{details}"
            )
            
            # Submit the log if run_logger is True
            await self.submit_logs(log_message)

    def get_logs(self):
        return self.logs