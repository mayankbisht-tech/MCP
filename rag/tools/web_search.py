from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

search = GoogleSerperAPIWrapper()


def search_web(query: str) -> str:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    return search.run(query)


@tool("serper_search")
def serper_search(query: str) -> str:
    return search_web(query)


@tool("brave_search")
def brave_search(query: str) -> str:
    return search_web(query)


if __name__ == "__main__":
    print(search_web("bitcoin price today"))
