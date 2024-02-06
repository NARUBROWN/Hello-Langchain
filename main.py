from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
# invoke: (v) 부르다

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 세계적인 수준의 기술 저자야"),
    ("user", "{input}")
])

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
# print(chain.invoke({"input": "어떻게 lansmith가 테스트를 도울 수 있어?"}))

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "무함마드 알리와 앨런 튜링 중에서 누가 더 오래 살았나요?",
        "answer": """
        여기에서 후속 질문이 필요한가요?: Yes.
        후속 질문: 무함마드 알리가 사망했을 때 몇 살이었나요?
        중간 답변: 무함마드 알리는 사망 당시 74세였습니다.
        후속 질문: 앨런 튜링은 사망했을 때 몇 살이었나요?
        중간 답변: 앨런 튜링은 사망 당시 41세였습니다.
        그래서 최종 답은 다음과 같습니다: 무함마드 알리
        """,
    },
    {
        "question": "크레이그리스트의 설립자는 언제 태어났나요?",
        "answer": """
        여기에서 후속 질문이 필요한가요? Yes.
        후속 질문: 크레이그리스트의 설립자는 누구인가요?
        중간 답변: 크레이그리스트는 크레이그 뉴마크가 설립했습니다.
        후속 질문: 크레이그 뉴마크는 언제 태어났나요?
        중간 답변: 크레이그 뉴마크는 1952년 12월 6일에 태어났습니다.
        그래서 최종 답은 다음과 같습니다: 1952년 12월 6일
        """,
    },
    {
        "question": "조지 워싱턴의 외할아버지는 누구였나요?",
        "answer": """
        여기에서 후속 질문이 필요한가요? Yes.
        후속 질문: 조지 워싱턴의 어머니는 누구였나요?
        중간 답변: 조지 워싱턴의 어머니는 메리 볼 워싱턴이었습니다.
        후속 질문: 메리 볼 워싱턴의 아버지는 누구였나요?
        중간 정답: 메리볼 워싱턴의 아버지는 조셉 볼입니다.
        그래서 최종 답은 다음과 같습니다: 조셉 볼
        """,
    },
    {
        "question": "죠스와 카지노 로얄의 감독은 모두 같은 나라 출신인가요?",
        "answer": """
        여기에 후속 질문이 필요한가요? Yes.
        후속 질문: 죠스의 감독은 누구인가요?
        중간 답변: 죠스의 감독은 스티븐 스필버그입니다.
        후속 질문: 스티븐 스필버그는 어디 출신인가요?
        중간 답변: 미국
        후속 질문: 카지노 로얄의 감독은 누구인가요?
        중간 답변: 카지노 로얄의 감독은 마틴 캠벨입니다.
        후속 질문: 마틴 캠벨은 어디 출신인가요?
        중간 답변: 뉴질랜드
        그래서 최종 답은 다음과 같습니다. 아니요
        """,
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)
