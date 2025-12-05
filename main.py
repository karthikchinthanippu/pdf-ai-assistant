# main.py

from langchain_core.messages import HumanMessage
from graph_app import build_app


def run_cli():
    app = build_app()
    messages = []

    print("\n📘 PDF Assistant Ready (Local LLM Enabled)")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break

        messages.append(HumanMessage(content=user))

        result = app.invoke({"messages": messages, "mode": "qa"})
        reply = result["messages"][-1].content

        print("\nAssistant:", reply, "\n")

        messages = result["messages"]


if __name__ == "__main__":
    run_cli()
