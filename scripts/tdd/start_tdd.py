import time, subprocess, os, click
from watchdog.observers import Observer  
from watchdog.events import PatternMatchingEventHandler


class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.py"]

    def __init__(self, source_path, test_path):
        self.source_path = source_path
        self.test_path = test_path

    def process(self, event):
        subprocess.run(['pytest', '--cov={}'.format(self.source_path), self.test_path])

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)

@click.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.argument('test_path', type=click.Path(exists=True))
def run_tdd(source_path, test_path):
    """
    Monitor the changes of the *.py files in SOURCE_PATH and TEST_PATH and run pytest with coverage support
    """
    observer = Observer()
    observer.schedule(MyHandler(source_path, test_path), os.getcwd(), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    run_tdd()


