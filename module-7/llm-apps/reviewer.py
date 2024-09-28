
from ai_scientist.perform_review import load_paper, perform_review
import openai
import typer
import agentops
from rich.console import Console
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow



def review_paper(paper_pdf_path: str, client: openai.OpenAI) -> str:
    model="gpt-4o-mini-2024-07-18"
    # Load paper from pdf file (raw text)
    paper_txt = load_paper(paper_pdf_path)
    review = perform_review(
        paper_txt,
        model,
        client,
        num_reflections=5,
        num_fs_examples=1,
        num_reviews_ensemble=5,
        temperature=0.1,
    )

    res = f'{review["Overall"]}\n{review["Decision"]}\n{review["Weaknesses"]}'
    return res


console = Console()


def run_pipeline():

    paper_pdf_path = "llm-apps/2408.06292v2_no_appendix.pdf"

    # console.print("1. Agentops", style="bold green")
    # agentops.init()
    # client_agentops = openai.Client()
    # result = review_paper(paper_pdf_path=paper_pdf_path, client=client_agentops)
    # agentops.end_session()

    # console.print("2. LangSmith", style="bold green")
    # client_lang_smith = wrap_openai(openai.Client())
    # result = review_paper(paper_pdf_path=paper_pdf_path, client=client_lang_smith)

    console.print("3. OpenllMetry", style="bold green")
    Traceloop.init(app_name="ai-scientist-v2")    
    client_traceloop = openai.Client()
    review_paper_traceloop = workflow(name="paper-review")(review_paper)
    result = review_paper_traceloop(paper_pdf_path=paper_pdf_path, client=client_traceloop)

    print(f"result = {result}")
if __name__ == "__main__":
    typer.run(run_pipeline)
