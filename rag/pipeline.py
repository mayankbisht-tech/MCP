from typing_extensions import Literal
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

from rag.model_config import GROQ_API_KEY, LLM_MODEL_NAME
from rag.tools.web_search import brave_search, serper_search

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


class Route(BaseModel):
    step: Literal["web_search", "retrieval"] = Field(
        None, description="The next step in the routing process"
    )


router = llm.with_structured_output(Route)


class State(TypedDict):
    input: str
    decision: str
    output: str
def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input question to web_search or retrieval based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


def route_decision(state: State):
    if state["decision"] == "web_search":
        return "web_search_node"
    elif state["decision"] == "retrieval":
        return "retrieval_node"


router_builder = StateGraph(State)

router_builder.add_node("web_search_node", web_search)
router_builder.add_node("retrieval_node", retrieval)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  
        "web_search": "web_search_node",
        "retrieval": "retrieval_node",

    },
)
router_builder.add_edge("web_search_node", END)
router_builder.add_edge("retrieval_node", END)
router_builder.add_edge("llm_call_router", END)

router_workflow = router_builder.compile()

display(Image(router_workflow.get_graph().draw_mermaid_png()))

state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])

def build_agent():
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Add it to your .env file or environment.")

    llm = init_chat_model(
        model=LLM_MODEL_NAME,
        model_provider="groq",
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    return create_agent(
        llm,
        [serper_search, brave_search],
        system_prompt=(
            "You are a helpful chatbot. Answer the user's question directly and naturally. "
            "Do not mention tools, function calls, or internal reasoning. "
            "If you do not need web search, answer from your own knowledge. "
            "Keep responses concise unless the user asks for detail."
        ),
    )


def chat_loop():
    agent = build_agent()
    conversation = []

    print("Type your questions. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = agent.invoke({"messages": conversation + [("user", question)]})
        answer = result["messages"][-1]
        print(f"Bot: {answer.content}")

        conversation.extend([("user", question), ("assistant", answer.content)])


def main():
    chat_loop()


if __name__ == "__main__":
    main()
