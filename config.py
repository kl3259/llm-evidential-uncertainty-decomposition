import torch

class CollectConfig:
    max_new_tokens: int = 50
    batch_size: int = 8
    temperature: float = 0.0         
    top_p: float = 1.0
    do_sample: bool = False         
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    judge_type: str = "rouge" 

class CollectConfigDebug:
    max_new_tokens: int = 50
    batch_size: int = 1
    temperature: float = 0.0         
    top_p: float = 1.0
    do_sample: bool = False         
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    judge_type: str = "rouge" 

class SESamplingConfig:
    max_new_tokens: int = 50
    batch_size: int = 1
    temperature: float = 1.0         
    top_p: float = 0.9
    do_sample: bool = True         
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    judge_type: str = "rouge" 

class SESamplingConfigDebug:
    max_new_tokens: int = 50
    batch_size: int = 1
    temperature: float = 1.0         
    top_p: float = 0.9
    do_sample: bool = True         
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    judge_type: str = "rouge" 

class ITPSamplingConfig:
    max_new_tokens: int = 50
    batch_size: int = 1
    temperature: float = 0.6
    top_p: float = 0.9
    do_sample: bool = True   
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    judge_type: str = "rouge" 


