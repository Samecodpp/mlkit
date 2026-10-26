import logging
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logging.getLogger(__name__).addHandler(logging.NullHandler())
