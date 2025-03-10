import dataclasses
import discopy.braided as db

@dataclasses.dataclass
class Generator:
    idx: int
    data: dict

    def __init__(self, idx: int, data: dict | None = None):
        self.idx = idx
        self.data = {} if data is None else data

    @property
    def sign(self):
        return 1 if self.idx > 0 else -1
    
    @property
    def pos(self):
        return abs(self.idx)

    def copy(self) -> "Generator":
        return Generator(self.idx, self.data.copy())
    
    def overlap(self, other: "Generator") -> bool:
        return abs(self.pos - other.pos) < 2

@dataclasses.dataclass
class Braid:
    gens: list[Generator]

    @property
    def strands(self) -> int:
        return max((w.pos + 1 for w in self.gens), default=0)
    
    def draw(self, highlight=(), **kwargs):
        dom = db.Ty(*['*']*self.strands)
        diag = db.Id(dom)

        for i, w in enumerate(self.gens):
            if w.idx > 0:
                c = db.Braid(db.Ty('*'), db.Ty('*')) 
            else:
                c = db.Braid(db.Ty('*'), db.Ty('*'))[::-1]
            if highlight is not None and i in highlight:
                c = db.Box(f"{i}: {w.idx}", db.Ty('*', '*'), db.Ty('*', '*'), color="red") >> c >> db.Box("", db.Ty('*', '*'), db.Ty('*', '*'), color="red")
            diag = diag >> db.Id(db.Ty(*['*']*(abs(w.idx) - 1))) @ c @ db.Id(db.Ty(*['*']*(self.strands - abs(w.idx) - 1)))
        
        diag.draw(**kwargs)
    
    @property
    def writhe(self) -> int:
        return sum(1 if g.idx > 0 else -1 for g in self.gens)
    
    @staticmethod
    def from_word(word: list[int]) -> "Braid":
        return Braid([Generator(i) for i in word])
 
    def to_word(self) -> list[int]:
        return [g.idx for g in self.gens]
    
    def __getitem__(self, idx: int) -> Generator:
        return self.gens[idx]
        
    def __setitem__(self, idx: int, value: Generator):
        self.gens[idx] = value
    
    def __delitem__(self, idx: int):
        del self.gens[idx]

    def __len__(self) -> int:
        return len(self.gens)
    
    def copy(self) -> "Braid":
        return Braid([g.copy() for g in self.gens])

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

phi = ((5 ** 0.5) + 1) / 2
