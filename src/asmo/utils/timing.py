"""
Decorator @timeit for timing running times and counting function calls

function print_timeit() for printing result

example use:

    from timing import timeit, print_timeit

    # Define function with decorator
    @timeit
    def blibli():
         s = 0
         for _ in range(1000000):
             s += _**2

    # Add decorator to already defined function
    blibli = timeit(blibli)

    ...
    blibli()
    blibli()
    blabla()

    print_timeit()

    >> RESULT
    _______________________________________________________
     blibli               :  100.4 seconds       2 calls
     blabla               :  130.0 seconds       1 calls
    _______________________________________________________
     Total time           :  150.0 seconds
    _______________________________________________________


"""

from functools import wraps
import time
import threading
import multiprocessing

# Define global dictionaries
TIME_dict: dict[str, float] = {}
COUNT_dict: dict[str, int] = {}
START_TIME: float = time.perf_counter()


def timeit(func, keyname=None):
    """timeit. A python decorator which to record future calls (time and call count) of the function func.

    Args:
        func: A function to decorate
        keyname: name in log - usefull for long names
    """
    global TIME_dict, COUNT_dict
    if keyname == None:
        keyname = func.__name__
    TIME_dict[keyname] = 0
    COUNT_dict[keyname] = 0

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        TIME_dict[keyname] += total_time
        COUNT_dict[keyname] += 1
        return result

    return timeit_wrapper


def print_timeit(tolerance=0, logger=None):
    """print_timeit. Prints the current TIME_DICT

    Args:
        tolerance: lower tolerance (seconds) for when a time entry should be printed
        logger: name of logging object to divert the print.
    """
    global TIME_dict
    out_str = "\n"
    hline = 70
    out_str += "\t" + "_" * hline + "\n"
    TIME_dict = {k: v for k, v in sorted(TIME_dict.items(), key=lambda item: item[1])}
    for k, v in TIME_dict.items():
        calls = COUNT_dict[k]
        # if v > 0.01:
        if calls != 0 and v > tolerance:
            calls = f"{calls:13.2e}" if calls > 1_000_000 else f"{calls:13}"
            # print(f" {k:27} : {v:10.2f} seconds {calls} calls")
            out_str += "\t" + f" {k:27} : {v:10.2f} seconds {calls} calls" + "\n"

    total_time = time.perf_counter() - START_TIME

    if total_time > tolerance:
        out_str += "\t" + "_" * hline + "\n"
        out_str += "\t" + f" {'Total time ':27} : {total_time:10.2f} seconds" + "\n"
        out_str += "\t" + "_" * hline + "\n"

        print(out_str)
        if logger:
            logger.info(out_str)

    return TIME_dict


def reset_timeit():
    """reset_timeit. Resets all time/count objects: TIME_dict, COUNT_dict and START_TIME."""
    global TIME_dict
    global COUNT_dict
    global START_TIME

    TIME_dict = {k: 0 for k in TIME_dict.keys()}
    COUNT_dict = {k: 0 for k in COUNT_dict.keys()}
    START_TIME = time.perf_counter()


def time_object(object_name, prefix=None):
    """
    Modifies the incoming object_name (class or module) by adding the timeit_wrapper to each callable in the object.

    exceptions include '__class__', '__new__', '__getattribute__' and names which include 'recursion'
    """
    if prefix == None:
        prefix = object_name.__name__

    for fct_str in dir(object_name):
        fct = getattr(object_name, fct_str)
        if (
            callable(fct)
            and fct_str not in ["__class__", "__new__", "__getattribute__"]
            and "recursion" not in fct_str
        ):
            fct = timeit(fct, f"{prefix}.{fct_str}")
            setattr(object_name, fct_str, fct)
    return object_name


def log_every_x_minutes(x, logger):
    """
    A decorator for creating a log every x minutes. Threading is required as the process has to run paralell to calling the function func


    example use:
        import logging
        logging.basicConfig(level=logging.INFO, filename=logname)
        logger = logging.getLogger(logname)

        @log_every_x_minutes(30, logger)
        def example_function_which_takes_a_long_time_to_run():
            time.sleep(a long time)
    (or)
        example_fct = log_every_x_minutes(30, logger)(example_fct)


    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            def log_time():
                elapsed_time = time.time() - start_time
                logger.info(
                    f"The function {func.__name__} has been running for {elapsed_time / 60:.2f} minutes"
                )
                # Schedule the next log after x minutes
                threading.Timer(x * 60, log_time).start()

            # Start the initial log
            log_time()

            # Call the original function
            result = func(*args, **kwargs)

            # The function has completed; no need to stop the timer
            return result

        return wrapper

    return decorator


def target(queue, func, *args, **kwargs):
    result = func(*args, **kwargs)
    queue.put(result)


def terminate_after_x_minutes(x: int, logger=None):
    """terminate_after_x_minutes. A decorator which exits a function after x minutes

    Args:
        x (int): x minutes
        logger: a logging object
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a queue to capture the return value
            queue = multiprocessing.Queue()

            # Start the function as a process
            p = multiprocessing.Process(
                target=target, args=(queue, func, *args), kwargs=kwargs
            )
            p.start()

            # Wait for the specified time or until the process finishes
            p.join(timeout=x * 60)

            # If the process is still active after the wait time
            if p.is_alive():
                message = f"{func.__name__} is running... killing process after {x:.2f} minutes"
                print(message)
                if logger:
                    logger.info(message)
                    logger.warning(message)

                # Terminate the process
                p.terminate()

                # Ensure the process has terminated
                p.join()
                return None
            else:
                # Get the result from the queue
                if not queue.empty():
                    return queue.get()
                return None

        return wrapper

    return decorator


def target(queue, func, *args, **kwargs):
    result = func(*args, **kwargs)
    queue.put(result)


def terminate_and_log(max_time: int, log_interval=0, logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # Create a queue to capture the return value
            queue = multiprocessing.Queue()

            # Start the function as a process
            p = multiprocessing.Process(
                target=target, args=(queue, func, *args), kwargs=kwargs
            )
            p.start()

            start_time = time.time()
            last_log_time = start_time
            check_interval = 1  # Check every second

            while time.time() - start_time < max_time * 60:
                time.sleep(check_interval)
                current_time = time.time()

                # Log every y minutes
                if log_interval and current_time - last_log_time >= log_interval * 60:
                    if p.is_alive():
                        elapsed_time = current_time - start_time
                        message = f"The function {func.__name__} has been running for {elapsed_time / 60:.2f} minutes"
                        print(f"{message}")
                        if logger:
                            logger.info(message)

                        last_log_time = current_time
                print(f"{p.is_alive()=}")
                if not p.is_alive():
                    break

            # If the process is still active after the wait time
            if p.is_alive():
                message = f"{func.__name__} is running... killing process after {max_time:.2f} minutes"
                print(message)
                if logger:
                    logger.info(message)
                    logger.warning(message)

                # Terminate the process
                p.terminate()

                # Ensure the process has terminated
                p.join()
                return None
            else:
                # Get the result from the queue
                if not queue.empty():
                    return queue.get()
                return None

        return wrapper

    return decorator


def set_defaults(**default_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key, value in default_kwargs.items():
                if key not in kwargs:
                    kwargs[key] = value
            return func(*args, **kwargs)

        return wrapper

    return decorator
