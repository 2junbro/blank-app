import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    # 1. Tavily Search Tool 호출하기
    return TavilySearchResults(k=6, name="web_search")

# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    # 2. PDF 로더 초기화 및 문서 불러오기
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 3. 텍스트를 일정 단위(chunk)로 분할하기
    #    - chunk_size: 한 덩어리의 최대 길이
    #    - chunk_overlap: 덩어리 간 겹치는 부분 길이
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    # 4. 분할된 문서들을 임베딩하여 벡터 DB(FAISS)에 저장하기
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})
    # 5. 검색기(retriever) 객체 생성
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # 6. retriever를 LangChain Tool 형태로 변환 -> name은 pdf_search로 지정
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="이 도구는 업로드된 pdf 문서에 직접 접근 할 수 있게 해줍니다 "
                    "질문이 pdf에서 답변할 수 있을때는 항상 이 도구를 먼저 사용하세요."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        # 7. 여러분의 챗봇에 맞는 system message 작성하기
        "한국 주식시장에 투자를 돕는 유용한 어시스턴트입니다. "
         "먼저 항상 pdf_serach 를 사용하세요 "
         "만약 pdf_search 에서 관련 결과가 없으면 즉시 web_search 만 호출하세요 "
         "두도구를 절대 섞어서 사용하지 마세요"
         "전문적이고 친근한 톤으로 한국어로 말하며 이모지를 포함하세요"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 8.agent 및 aagent_executor 생성하기
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수 (툴 사용 내역 제거)
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # 9. intermediate_steps 통해 사용툴을 출력할 수 있는 코드 완성하기
    # intermediate_steps에서 마지막만 가져오기
    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"


# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    # 10. 여러분의 챗봇에 맞는 스타일로 변경하기
    st.set_page_config(page_title="한국주식시장 동향 챗봇", layout="wide", page_icon="🤖")
    
    #st.image('data/동학개미.jpg', width=800)
    with st.container():
        col1, col2 = st.columns([1, 1])  # 비율 조정 가능 예: [2,1]
        with col1:
            st.image('data/한국주식.jpg', use_container_width=True, caption="이미지 1")
        with col2:
            st.image('data/동학개미.jpg', use_container_width=True, caption="이미지 2")

    st.markdown('---')
    st.title("안녕하세요! RAG + Web을 활용한 '한국주식시장 동향 챗봇' 입니다")  

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")
        pdf_docs = st.file_uploader("PDF 파일 업로드", accept_multiple_files=True)

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web()]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("질문을 입력하세요")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()
