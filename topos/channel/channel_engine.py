# channel_engine.py

import asyncio
import traceback


class ChannelEngine:
    """
    A generalized engine for managing asynchronous tasks in a queue.
    """

    def __init__(self):
        """
        Initializes the Engine with an asynchronous task queue and sets it to a non-running state.
        """
        print(f"\t[ Engine :: init ]")
        self.task_queue = asyncio.Queue()
        self.task_handlers = {}
        self.running = False
        self.processing_task = None
        self._lock = asyncio.Lock()

    def register_task_handler(self, task_type, handler):
        """
        Registers a handler function for a specific task type.

        Args:
            task_type: A string identifier for the task type.
            handler: An asynchronous function that will handle tasks of the given type.
        """
        self.task_handlers[task_type] = handler

    async def add_task(self, task):
        """
        Adds a task to the task queue and potentially starts task processing if the engine is not already running.

        Args:
            task: A dictionary representing the task with a 'type' key and additional task-specific data.
        """
        print(f"\t[ Engine :: Adding task to queue: {task['type']} ]")
        await self.task_queue.put(task)
        print(f"\t\t[ Engine :: Task added to queue: {task['type']} ]")

        if not self.running:
            print(f"\t[ Engine :: Starting task processing ]")
            await asyncio.sleep(0)  # Yield control to the event loop
            await self.start_processing()

    async def wait_for_tasks(self):
        """
        Waits for all tasks in the queue to be completed.
        """
        print(f"\t[ Engine :: Waiting for all tasks to complete ]")
        await self.task_queue.join()

    async def reset_processing_queue(self):
        """
        Resets the processing queue by stopping current processing, canceling any running task,
        clearing the queue, and resetting the 'running' flag.
        """
        print(f"\t[ Engine :: Resetting processing queue ]")
        async with self._lock:
            self.running = False
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self.current_generation = None
            self.running = True
            self.processing_task = asyncio.create_task(self.process_tasks())

    async def start_processing(self):
        """
        Starts processing tasks from the queue by setting the 'running' flag to True and creating
        a task to handle the processing.
        """
        print(f"\t[ Engine :: Starting task processing ]")
        self.running = True
        self.processing_task = asyncio.create_task(self.process_tasks())

    async def stop_processing(self):
        """
        Stops task processing by setting the 'running' flag to False and canceling any running task.
        """
        print(f"\t[ Engine :: Stopping task processing ]")
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    async def process_tasks(self):
        """
        Continuously processes tasks from the queue as long as the 'running' flag is True.
        Handles task execution, completion, and potential errors.
        """
        print(f"\t[ Engine :: Starting to process tasks ]")
        while self.running:
            try:
                task = await self.task_queue.get()
                print(f"\t\t[ Engine :: Processing task: {task['type']} ]")
                try:
                    await self.execute_task(task)
                    print(f"\t\t[ Engine :: Finished processing task: {task['type']} ]")
                finally:
                    self.task_queue.task_done()
            except asyncio.CancelledError:
                print(f"\t[ Engine :: Task processing was cancelled ]")
                break
            except Exception as e:
                print(f"\t[ Engine :: Error processing task: {e} ]")
                traceback.print_exc()
        print(f"\t[ Engine :: Stopped processing tasks ]")

    async def execute_task(self, task):
        """
        Executes a task by looking up its handler in the task_handlers dictionary
        and passing the relevant task data as arguments to the handler.

        Args:
            task: A dictionary representing the task with a 'type' key and additional task-specific data.
        """
        task_type = task['type']
        handler = self.task_handlers.get(task_type)
        if handler:
            print(f"\t\t[ Engine :: Executing task: {task_type} ]")
            try:
                task_data = task.copy()  # Create a copy of the task data
                task_data.pop('type')  # Remove the 'type' key
                await handler(**task_data)  # Pass task data (without 'type') as keyword arguments
                print(f"\t\t[ Engine :: Finished processing task: {task_type} ]")
            except Exception as e:
                print(f"\t\t[ Engine :: Error processing task: {e} ]")
                traceback.print_exc()
        else:
            print(f"\t\t\t[ Engine :: No handler registered for task type: {task_type} ]")

