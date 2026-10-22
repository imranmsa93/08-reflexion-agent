import sys

from graph import graph

DEFAULT_QUESTION = "What does LangGraph persistence do?"


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    result = graph.invoke({"question": question})

    print("QUESTION:", question)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for url in result["references"]:
        print(" -", url)
    print("\nREVISIONS:", result["rounds"])


if __name__ == "__main__":
    main()