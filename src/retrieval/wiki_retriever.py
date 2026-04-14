from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import requests

from src.retrieval.base_retriever import BaseRetriever


class WikipediaRetriever(BaseRetriever):
    """
    Wikipedia API 기반 검색 리트리버 (ko.wikipedia.org).

    - build_index: 사용하지 않음
    - retrieve: 단일 질의(str)에 대해 상위 문서 요약 반환
    """

    def __init__(
        self,
        lang: str = "ko",
        top_k: int = 3,
        timeout: int = 10,
        user_agent: str = "Ko-CSAT-Reasoning/0.1 (contact: research@example.com)",
        **kwargs: Any,
    ):
        super().__init__(data_path=f"https://{lang}.wikipedia.org", **kwargs)
        self.lang = lang
        self.top_k = top_k
        self.timeout = timeout
        self.user_agent = user_agent
        self.api_base = f"https://{lang}.wikipedia.org/w/api.php"

    def build_index(self):
        return

    def retrieve(self, query_or_dataset: Union[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not isinstance(query_or_dataset, str):
            raise ValueError("WikipediaRetriever.retrieve supports a single query string.")

        k = top_k or self.top_k
        search_results = self._search(query_or_dataset, k)
        if not search_results:
            return []

        page_ids = [str(r["pageid"]) for r in search_results]
        pages = self._fetch_pages(page_ids)

        results: List[Dict[str, Any]] = []
        for r in search_results:
            pid = str(r["pageid"])
            page = pages.get(pid, {})
            title = page.get("title", r.get("title"))
            extract = page.get("extract") or ""
            url = page.get("fullurl") or ""

            results.append(
                {
                    "doc_id": pid,
                    "title": title,
                    "score": r.get("score"),
                    "content": extract,
                    "metadata": {
                        "full_content": extract,
                        "url": url,
                        "pageid": pid,
                    },
                }
            )

        return results

    def _search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srlimit": top_k,
            "srsearch": query,
        }
        headers = {"User-Agent": self.user_agent}
        resp = requests.get(self.api_base, params=params, timeout=self.timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("query", {}).get("search", []) or []

    def _fetch_pages(self, page_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not page_ids:
            return {}
        params = {
            "action": "query",
            "pageids": "|".join(page_ids),
            "format": "json",
            "prop": "extracts|info",
            "explaintext": 1,
            "exintro": 1,
            "inprop": "url",
        }
        headers = {"User-Agent": self.user_agent}
        resp = requests.get(self.api_base, params=params, timeout=self.timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("query", {}).get("pages", {}) or {}


__all__ = ["WikipediaRetriever"]
