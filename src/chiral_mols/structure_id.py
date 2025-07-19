from pydantic import BaseModel

class StructureID(BaseModel, validate_assignment=True):
    StructureID: int
    MoleculeID: int
    EnantiomerID: int
    ConformerID: int

    model_config = dict(frozen=True)

    def to_string(self) -> str:
        return (
            f"{self.StructureID}-"
            f"{self.MoleculeID}-"
            f"{self.EnantiomerID}-"
            f"{self.ConformerID}"
        )

    @classmethod
    def from_string(cls, s: str) -> "StructureID":
        try:
            st, mol, en, conf = map(int, s.split("-"))
        except ValueError:
            raise ValueError(f"Invalid StructureID string: {s!r}") from None
        return cls(
            StructureID=st,
            MoleculeID=mol,
            EnantiomerID=en,
            ConformerID=conf,
        )