from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

from langchain.chains import RetrievalQA
import os
import magic
import nltk

"""
!!!!!!!!!!!运行前注意要配置网络!!!!!!!!!!!
!!!!!!!!!!!运行前注意要配置网络!!!!!!!!!!!
!!!!!!!!!!!运行前注意要配置网络!!!!!!!!!!!
"""

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
# 这样可以将这个 API 设置为环境变量

"""
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain import ConversationChain

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
"""

# nltk.download('punkt')

# TODO: 0. 实例化
# 利用 openai 的 API初始化一个大模型, 注意网络配置, 否则会出现 APIConnectionError的报错
print("Start to initialize the LLM...")
llm = OpenAI(temperature=0.7)
print("Initialize the LLM successfully!")
'''{
# TODO: 1. Test of LLM
"""
# SIMPLEST example to use the LLM
text = "What are 5 vacation destinations for someone who like to eat pasta?"
print(llm(text))


EXAMPLE_OUTPUT:
1. Rome, Italy
2. Tuscany, Italy
3. Naples, Italy
4. Venice, Italy
5. Sicily, Italy
"""

# TODO: 2. Templates: Manage prompts for LLMs
"""
# NEED to import PromptTemplate form langchain.prompts

prompt = PromptTemplate(
    input_variables=["FOOD_NAME"],
    template="what are 5 vacation destinations for someone who like to eat {FOOD_NAME}?"
)
print("Initialize the prompts")
print(prompt.format(FOOD_NAME="desert"))
print(llm(prompt.format(FOOD_NAME="desert")))


EXAMPLE_OUTPUT:
what are 5 vacation destinations for someone who like to eat desert?


1. Dubai, United Arab Emirates
2. Marrakech, Morocco
3. Tucson, Arizona
4. Santa Fe, New Mexico
5. San Diego, California
"""

# TODO: 3. Chain: Combine LLMs and prompts in multi-step workflows
"""
# NEED to import LLMChain form langchain.chains

prompt = PromptTemplate(
    input_variables=["FOOD_NAME"],
    template="what are 5 vacation destinations for someone who like to eat {FOOD_NAME}?"
)
print("Start to run the prompt...")
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("fruit"))

EXAMPLE_OUTPUT:
1. Hawaii 
2. Costa Rica 
3. Mauritius 
4. Thailand 
5. Bali
"""

# TODO: 4. Agents: Dynamically call chains based on user input
# NEED: `pip install google-search-results` and a SerpAPI to use
"""
os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_API_KEY"

# Initialize an agent with:
#     1. The tools
#     2. The LLM
#     3. The type of agent you want to use (default: zero-shot-react-description)


tools = load_tools(["serpapi", "llm-math"], llm=llm)
tiny_math_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

tiny_math_agent.run("Who is the current leader of japan? What is the largest prime number that is smaller than their age.")


EXAMPLE_OUTPUT:
> Entering new AgentExecutor chain...
 I need to first find the leader's age.
Action: Search
Action Input: "Current leader of Japan"
Observation: Fumio Kishida is the current prime minister of Japan, replacing Yoshihide Suga on 4 October 2021. As of 27 August 2023, there have been 64 individual prime ministers serving 101 terms of office.
Thought: I need to use a calculator to figure out the largest prime number that is smaller than the current leader's age.
Action: Calculator
Action Input: 64
Observation: Answer: 64
Thought: I now know the final answer.
Final Answer: 63

> Finished chain.

# 这里会有 幻觉 和 Retrying 每次的输出结果可能不一样 有 61 63
"""

# TODO: 5. Memory: Add state to chains and agents
"""
# NEED import ConversationChain
conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input="Hi there!")
conversation.predict(input="I'm doing well! Just having a conversation with an AI")
conversation.predict(input="what was the first thing I said to you?")
conversation.predict(input="what is an alternative phrase for the first thing I said to you?")
print("--DONE--")

> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi there!
AI:  Hello there! It's nice to meet you. How can I help you today?
Human: I'm doing well! Just having a conversation with an AI
AI:  Great! I'm glad to hear that. It's always nice to chat with someone. What would you like to talk about?
Human: what was the first thing I said to you?
AI:  You said, "Hi there!"
Human: what is an alternative phrase for the first thing I said to you?
AI:  You could have said, "Greetings!" or "Hello!"

# 这个有其实很慢 在 jupyter notebook 中或许会有更好的效果
"""

}'''

print("start documents loading...")
loader = DirectoryLoader('FOLDER_LOCATION', glob='**/*.txt')
# 不要用中文字符写地址!!!!!!!!!
documents = loader.load()
print("documents loaded successfully!")

print("initialize the text splitter...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
print("initialize successfully!")

print("start text splitting...")
texts = text_splitter.split_documents(documents)
print("splitting successfully!")


print("start embedding...")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
print("embedding successfully...")

print("start vectorization...")
docsearch = Chroma.from_documents(texts, embeddings)
"""
qa = VectorDBQA.from_chain_type(llm=llm,
                                chain_type='stuff',
                                vectorstore=docsearch)
"""
qa = VectorDBQA.from_chain_type(llm=llm,
                                chain_type='stuff',
                                vectorstore=docsearch,
                                return_source_documents=True)


print("vectorize successfully!")

print("------------------------------------------------------------------------------------")
query = '什么是小坝镇基层应急指挥'
print("Me:" + query)
# qa.run(query)

result = qa({"query": query})

print("------------------------------------------------------------------------------------")
print("answer:")
print(result["result"])
print("------------------------------------------------------------------------------------")
print(result['source_documents'])
print("------------------------------------------------------------------------------------")

