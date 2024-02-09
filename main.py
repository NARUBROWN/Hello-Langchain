from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
# invoke: (v) 부르다

from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_messages([
# ("system", "너는 세계적인 수준의 기술 저자야"),
# ("user", "{input}")
# ])

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

# chain = prompt | llm | output_parser
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

# 입력과 예시의 유사도에 따라 몇가지의 예제를 선택해주는 선택기
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# 예시들의 위치를 저장할 벡터 저장소
from langchain_community.vectorstores import Chroma
# 문자들을 벡터로 변환해주는 임베딩 클래스
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 선택기에 의해 선택될 수 있는 예제의 목록들
    examples,
    # 의미적 유사성을 측정하기 위해 사용되는 임베딩을 생성하는 임베딩 클래스입니다.
    OpenAIEmbeddings(),
    # 임베딩을 저장하고 유사성 검색을 수행하는 데 이용되는 벡터 저장소입니다.
    Chroma,
    # 생성할 예제의 개수
    k=1
)

# 입력한 내용과 가장 유사한 예시를 선택합니다.
question = input("메리 볼 워싱턴의 아버지는 누구니?")
selected_examples = example_selector.select_examples({"question": question})

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

chain = prompt | llm | output_parser

# print(chain.invoke({"input": "메리 볼 워싱턴의 아버지는 누구야?"}))


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


class Response(BaseModel):
    question: str = Field(description="여기에는 사용자의 질문이 들어가야해")
    answer: str = Field(description="여기에는 너의 답변이 들어가야해")


# 구문 분석기 설정 + 프롬프트 템플릿에 명령어 삽입
parser = JsonOutputParser(pydantic_object=Response)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}\n{format_instructions}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

print(example_selector.from_examples({{"input": "저녁 8시에 친구들과 함께 저녁 식사 약속이 있어."}}))

print(chain.invoke({"input": "메리 볼 워싱턴의 아버지는 누구야?"}))
