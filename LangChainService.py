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

import json
from pathlib import Path

file_path = 'corrected_detailed_events.json'
example = json.loads(Path(file_path).read_text(encoding='UTF8'))


class LangChainResponse(BaseModel):
    memberId: int = Field(description="""Include in prefix 'member_id' in the memberId""")
    categoryId: int = Field(description="Include in prefix 'category_id' in categoryId")
    title: str = Field(description="""In the title, include a one- or two-line summary sentence 
                                      that summarizes in the suffix 'user_dialog'""")
    contents: str = Field(description="Take notes summarizing the user's sentence. If there is no content, output null "
                                      "<important!>None is null</important> ")
    deadline: datetime = Field(description="Inside the deadline, you can infer the deadline as a date and time "
                                           "based on the date in 'current_time'. ")


model = ChatOpenAI(temperature=0)

parser = JsonOutputParser(pydantic_object=LangChainResponse)

example_prompt = PromptTemplate(
    template="{user_dialog}\n{answer}",
    input_variables=["user_dialog", "answer"]
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    example,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

memberId = 1
categoryId = 1

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="user_dialog: {user_dialog}",
    prefix="{format_instructions}, {current_time}, {member_id}, {category_id}",
    input_variables=["user_dialog"],
    partial_variables={"format_instructions": parser.get_format_instructions(),
                       "member_id": memberId,
                       "category_id": categoryId,
                       "current_time": str(datetime.now())}
)

chain = prompt | model | parser

#print(prompt.format(user_dialog= "친구네 집에서 아침 7시에만나서 아침을 같이 먹기로 했어"))

#selected_examples = example_selector.select_examples({"user_dialog": "친구네 집에서 아침 7시에만나서 아침을 같이 먹기로 했어"})
#print(selected_examples)
print(chain.invoke({
    "user_dialog": "user_dialog: 친구네 집에서 아침 7시에만나서 아침을 같이 먹기로 했어"
}))
