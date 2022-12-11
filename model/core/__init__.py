from .abstract import BaseModel
from .native import GenerativeLM as NativeModel
from .huggingface import GenerativeLM as HFModel
from .colossal import GenerativeLM as ColAIModel
from .pipeline import GenerativeLM as PPModel
