import requests
from typing import Any
import xml.etree.ElementTree as ET

from agent.state import PaperRecord

BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils"
)

HEADERS = {
    "User-Agent": (
        "biomedical-agent/0.1 "
        "(contact: fake_email@123.com)"
    )
}

def search_pubmed(
    query: str,
    retmax: int = 5
) -> list[str]:

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",

        "tool": "biomedical-agent",
        "email": "fake_email@123.com",
    }

    response = requests.get(
        f"{BASE_URL}/esearch.fcgi",
        params=params,
        headers = HEADERS,
        timeout=20,
        proxies={"http": None, "https": None}
    )

    response.raise_for_status()

    data = response.json()

    return data[
        "esearchresult"
    ]["idlist"]

def parse_pubmed_xml(
    xml_text: str
) -> list[PaperRecord]:

    root = ET.fromstring(
        xml_text
    )

    results = []

    for article in root.findall(
        ".//PubmedArticle"
    ):

        pmid = article.findtext(
            ".//PMID"
        )

        title = article.findtext(
            ".//ArticleTitle"
        )

        journal = article.findtext(
            ".//Journal/Title"
        )

        abstract_parts = []

        for node in article.findall(
            ".//Abstract/AbstractText"
        ):
            if node.text:
                abstract_parts.append(
                    node.text
                )

        abstract = " ".join(
            abstract_parts
        ) or None

        year = None

        year_text = article.findtext(
            ".//PubDate/Year"
        )

        if year_text:
            try:
                year = int(
                    year_text
                )
            except ValueError:
                pass

        authors = []

        for author in article.findall(
            ".//Author"
        ):
            lastname = author.findtext(
                "LastName"
            )

            initials = author.findtext(
                "Initials"
            )

            if lastname:
                if initials:
                    authors.append(
                        f"{lastname} {initials}"
                    )
                else:
                    authors.append(
                        lastname
                    )

        results.append(
            PaperRecord(
                pmid=pmid,
                title=title or "",
                abstract=abstract,
                year=year,
                journal=journal,
                authors=authors
            )
        )

    return results

def fetch_paper(
    pmids: list[str]
) -> list[PaperRecord]:

    if not pmids:
        return ""

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",

        "tool": "biomedical-agent",
        "email": "fake_email@123.com",
    }

    url = f"{BASE_URL}/efetch.fcgi"
    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=(10, 60),
            proxies={"http": None, "https": None}
        )

        response.raise_for_status()

        return parse_pubmed_xml(response.text)
    
    except requests.RequestException as e:

        raise RuntimeError(
            f"PubMed request failed: {e}"
        ) from e
