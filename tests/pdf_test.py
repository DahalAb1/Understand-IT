def add(a: int, b: int) -> int:
    return a + b

def test_add():
    assert add(2, 3) == 5

from app.utils import chunk_text 


def overlap_test():
    text = "Sentence. "*40 
    chunks = chunk_text(text)
    assert len(chunks) > 1 


from time import perf_counter
from app.utils import reword_text 

def cache_speed_test():
    txt = 'Hello Cache'*20
    t1 = perf_counter(); reword_text(txt,"Basic");t2 = perf_counter()
    reword_text(txt,'Basic');t3 = perf_counter()
    
    #expecting second call from theto be five times faster
    assert (t3-t2)*5 < (t1 -t0)

