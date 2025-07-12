import os
import requests
from functools import wraps
from typing import Annotated
import openai
from dotenv import load_dotenv
load_dotenv()


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
RAG_API = 'https://container-finrobot.dc4gg5b1dous0.eu-west-3.cs.amazonlightsail.com'


def get_annual_report_section(
    ticker_symbol: Annotated[str, "ticker symbol"],
    fyear: Annotated[int, "fiscal year of the annual report"],
    section: Annotated[
        str,
        "Section of the Annual Report [1, 2]",
    ],
) -> str:
    """
    Get a specific section of the Annual Report RAG API.
    """

    sections_dict = {
        'MRF': {
            '1': 'MANAGEMENT DISCUSSION AND ANALYSIS',
            '2': "BOARD'S REPORT",
            '3': "BALANCE SHEET",
            '4':"STATEMENT OF PROFIT AND LOSS",
            '5': "NOTES FORMING PART OF THE FINANCIAL STATEMENTS",
            '6': "CONSOLIDATED FINANCIAL STATEMENTS"
        },

        'DMART': {
            '1':  'Management Discussion and Analysis',
            '2': "Directors’ Report",
            '3': "Standalone Balance Sheet",
            '4': "Statement of Standalone Profit and Loss",
            '5': "Statement of Standalone Cash Flows"
        }
    }

    section_title = sections_dict[ticker_symbol][str(section)]

    payload ={
        "ticker":ticker_symbol,
        "year": fyear,
        "section": section_title
    }

    response = requests.post(f'{RAG_API}/extract_section', json=payload)
    response.raise_for_status()
    section_text = response.json()

    return section_text


def perform_similarity_search(
    ticker_symbol: Annotated[str, "ticker symbol"],
    fyear: Annotated[int, "fiscal year of the annual report"],
    section: Annotated[
        str,
        "Section of the Annual Report [1, 2, 3, 4, 5]",
    ],
    question: Annotated[
        str,
        "question or prhase to look by embeddings similarity search",
    ]
) -> str:
    """
    Get a specific section of the Annual Report RAG API and perform similarity search based on a question.
    """

    sections_dict = {
        'MRF': {
            '1': 'MANAGEMENT DISCUSSION AND ANALYSIS',
            '2': "BOARD'S REPORT",
            '3': "BALANCE SHEET",
            '4':"STATEMENT OF PROFIT AND LOSS",
            '5': "NOTES FORMING PART OF THE FINANCIAL STATEMENTS",
            '6': "CONSOLIDATED FINANCIAL STATEMENTS"
        },

        'DMART': {
            '1':  'Management Discussion and Analysis',
            '2': "Directors’ Report",
            '3': "Standalone Balance Sheet",
            '4': "Statement of Standalone Profit and Loss",
            '5': "Statement of Standalone Cash Flows"
        }
    }

    section_title = sections_dict[ticker_symbol][str(section)]

    payload ={
        "ticker":ticker_symbol,
        "year": fyear,
        "section": section_title,
        "question": question,
        "k": 5
    }

    response = requests.post(f'{RAG_API}/query', json=payload)
    response.raise_for_status()
    section_text = response.json()

    return section_text