from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# 입력과 예시의 유사도에 따라 몇가지의 예제를 선택해주는 선택기
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# 예시들의 위치를 저장할 벡터 저장소
from langchain_community.vectorstores import Chroma
# 문자들을 벡터로 변환해주는 임베딩 클래스
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import load_prompt

import json
from pathlib import Path

file_path = 'corrected_detailed_events.json'
data = json.loads(Path(file_path).read_text())

class LangChainResponse(BaseModel):
    memberId: int = Field(description="Include in prefix 'memberId' in the memberId")
    categoryId: int = Field(description="Include in prefix 'categoryId' in categoryId")
    title: str = Field(description="In the title, include a one- or two-line summary sentence that summarizes the "
                                   "user's answer.")
    contents: str = Field(description="Take notes summarizing the user's sentence. If there is no content, output null "
                                      "<important!>None is null</important> ")
    deadline: datetime = Field(description="Inside the deadline, you can infer the deadline as a date and time "
                                           "based on the date in 'nowTime'. ")


examples = [
    {
        "userDialog": "오늘 저녁 7시에 친구들이랑 만나서 놀기로 했어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "친구들과의 만남",
            "content": null,
            "deadline": "2024-02-07T19:00:00"
        """
    },
    {
        "userDialog": "내일 오전 7시에 부모님과 함께 러닝하기로 했어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "부모님과의 러닝",
            "content": null,
            "deadline": "2024-02-08T07:00:00"
        """
    },
    {
        "userDialog": "내일 오전 9시까지 무한상사에 파일을 보내야 해",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "무한상사에 파일 전송",
            "content": null,
            "deadline": "2024-02-08T09:00:00"
        """
    },
    {
        "userDialog": "모레 오후 6시에 책 읽기 모임이 있어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "책 읽기 모임",
            "content": null,
            "deadline": "2024-02-09T18:00:00"
        """
    },
    {
        "userDialog": "이번 주 금요일 밤 11시까지 개인 프로젝트 마감일이야",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "개인 프로젝트 마감일",
            "content": null,
            "deadline": "2024-02-10T23:00:00"
        """
    },
    {
        "userDialog": "다음 주 월요일 오전 10시에 건강 검진 예약이 있어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "건강 검진 예약",
            "content": null,
            "deadline": "2024-02-11T10:00:00"
        """
    },
    {
        "userDialog": "다음 주 화요일 밤 8시에 온라인 코딩 강의 듣기로 했어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "온라인 코딩 강의",
            "content": null,
            "deadline": "2024-02-12T20:00:00"
        """
    },
    {
        "userDialog": "다음 주 수요일 저녁 7시에 가족들과 저녁 식사 약속이 있어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "가족 저녁 식사",
            "content": null,
            "deadline": "2024-02-13T19:00:00"
        """
    },
    {
        "userDialog": "다음 주 목요일 오전 11시에 새로운 헬스클럽에 등록하기로 했어",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "헬스클럽 등록",
            "content": null,
            "deadline": "2024-02-14T11:00:00"
        """
    },
    {
        "userDialog": "다음 주 금요일 밤 9시까지 여행 계획을 세워야 해",
        "answer": """
            "memberId": "memberId",
            "categoryId": "categoryId",
            "title": "여행 계획 세우기",
            "content": null,
            "deadline": "2024-02-15T21:00:00"
        """
    }
]

model = ChatOpenAI(temperature=2)

parser = JsonOutputParser(pydantic_object=LangChainResponse)

example_prompt = PromptTemplate(
    template="{userDialog}\n{answer}",
    input_variables=["userDialog", "answer"]
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    data,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

memberId = 1
categoryId = 1

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="{userDialog}",
    prefix="{format_instructions}, {nowTime}, {memberId}, {categoryId}",
    input_variables=["userDialog"],
    partial_variables={"format_instructions": parser.get_format_instructions(),
                       "memberId": memberId,
                       "categoryId": categoryId,
                       "nowTime": str(datetime.now())}
)

chain = prompt | model | parser

selected_examples = example_selector.select_examples({"userDialog": "내일 저녁에 가족들이랑 신년회 파티를 열기로 했어"})
print(selected_examples)
print(chain.invoke({
    "userDialog": "내일 저녁에 가족들이랑 신년회 파티를 열기로 했어"
}))
