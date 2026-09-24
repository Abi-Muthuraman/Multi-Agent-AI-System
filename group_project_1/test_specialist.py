from specialist_agent import SpecialistAgent


agent = SpecialistAgent()

test_cases = [
    "I forgot my password and cannot log into my account.",
    "My laptop won't connect to Wi-Fi.",
    "My keyboard has stopped working.",
    "I cannot install the software I need.",
    "My company email is not synchronizing."
]


for i, question in enumerate(test_cases, start=1):

    print("\n" + "=" * 60)
    print(f"TEST CASE {i}")
    print("=" * 60)

    print("Question:")
    print(question)

    result = agent.answer(question)

    print("\nCategory:")
    print(result["category"])

    print("\nResolution:")
    print(result["resolution"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)