import os


def log(text):
    """Log the given text to the console."""
    if os.getenv("DEBUG", "False") == "True":
        print(f" \033[90m{text}\033[0m")  # it will print the text in gray color
