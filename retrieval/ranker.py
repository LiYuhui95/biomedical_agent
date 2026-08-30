from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from agent.state import PaperRecord


class SemanticRanker:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            local_files_only=True,
        )

    @staticmethod
    def paper_to_text(
        paper: PaperRecord
    ) -> str:

        title = paper.title or ""
        abstract = paper.abstract or ""

        return f"{title}. {abstract}"

    def rank(
        self,
        question: str,
        papers: list[PaperRecord],
        top_k: int = 5,
    ) -> list[tuple[PaperRecord, float]]:

        if not papers:
            return []

        question_embedding = self.model.encode(
            question,
            convert_to_tensor=True,
        )

        paper_texts = [
            self.paper_to_text(paper)
            for paper in papers
        ]

        paper_embeddings = self.model.encode(
            paper_texts,
            convert_to_tensor=True,
        )

        similarities = cos_sim(
            question_embedding,
            paper_embeddings,
        )[0]

        ranked = sorted(
            zip(
                papers,
                similarities.tolist(),
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]