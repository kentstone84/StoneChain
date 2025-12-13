# Migrating from LangChain to StoneChain

This guide shows how to migrate your LangChain code to StoneChain.

## Basic Completion

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
messages = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello!")
]
response = llm.invoke(messages)
print(response.content)
```

### StoneChain
```python
from stonechain import Anthropic, Message

llm = Anthropic()
response = llm.complete([
    Message.system("You are helpful."),
    Message.user("Hello!")
])
print(response.content)

# Or even simpler:
print(llm("Hello!"))
```

## Chains

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic()

prompt1 = ChatPromptTemplate.from_template("Analyze: {input}")
prompt2 = ChatPromptTemplate.from_template("Summarize: {analysis}")

chain = (
    {"input": lambda x: x}
    | prompt1
    | llm
    | StrOutputParser()
    | (lambda x: {"analysis": x})
    | prompt2
    | llm
    | StrOutputParser()
)

result = chain.invoke("AI trends")
```

### StoneChain
```python
from stonechain import Anthropic, Chain

llm = Anthropic()

chain = Chain(llm)
chain.add("analyze", "Analyze: {input}", "analysis")
chain.add("summarize", "Summarize: {analysis}", "summary")

result = chain.run(input="AI trends")
print(result["outputs"]["summary"])
```

## Agents

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain import hub

llm = ChatAnthropic()

tools = [
    Tool(
        name="calculator",
        func=lambda x: eval(x),
        description="Do math"
    )
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "What is 15 * 23?"})
```

### StoneChain
```python
from stonechain import Anthropic, Agent, Tool

llm = Anthropic()

tools = [
    Tool("calculator", "Do math", {"expression": {"type": "string"}}, lambda expression: eval(expression))
]

agent = Agent(llm, tools)
result = agent.run("What is 15 * 23?")
print(result["answer"])
```

## RAG

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

llm = ChatAnthropic()
embeddings = OpenAIEmbeddings()

loader = TextLoader("docs.txt")
docs = loader.load()

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = qa.invoke("What is StoneChain?")
```

### StoneChain
```python
from stonechain import Anthropic, RAG, Document

llm = Anthropic()
rag = RAG(llm)

rag.add([
    Document("StoneChain is a zero-dependency LLM framework."),
    Document("It was created by Kent Stone in 2025."),
])

result = rag.query("What is StoneChain?")
print(result["answer"])
```

## Conversation Memory

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatAnthropic()
memory = ConversationBufferMemory()

chain = ConversationChain(llm=llm, memory=memory)

chain.invoke({"input": "Hello!"})
chain.invoke({"input": "What did I just say?"})
```

### StoneChain
```python
from stonechain import Anthropic, Conversation

llm = Anthropic()
conv = Conversation(llm)

conv.chat("Hello!")
conv.chat("What did I just say?")
```

## Router

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

llm = ChatAnthropic()

code_prompt = ChatPromptTemplate.from_template("Write code: {input}")
math_prompt = ChatPromptTemplate.from_template("Solve: {input}")
default_prompt = ChatPromptTemplate.from_template("{input}")

branch = RunnableBranch(
    (lambda x: "code" in x["input"].lower(), code_prompt | llm),
    (lambda x: any(c.isdigit() for c in x["input"]), math_prompt | llm),
    default_prompt | llm
)

result = branch.invoke({"input": "Write python code"})
```

### StoneChain
```python
from stonechain import Anthropic, Router, Chain

llm = Anthropic()

code_chain = Chain(llm)
code_chain.add("code", "Write code: {input}", "result")

math_chain = Chain(llm)
math_chain.add("math", "Solve: {input}", "result")

router = Router(default=llm)
router.add("code", lambda x: "code" in x.lower(), code_chain)
router.add("math", lambda x: any(c.isdigit() for c in x), math_chain)

result = router.route("Write python code")
```

## Parallel Execution

### LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel

llm = ChatAnthropic()

parallel = RunnableParallel(
    analysis=llm,
    summary=llm,
)

# Complex setup required...
```

### StoneChain
```python
from stonechain import Anthropic, Parallel

llm = Anthropic()

results = Parallel.run([
    (llm, "Analyze AI"),
    (llm, "Summarize AI"),
])
```

## Key Differences

| Concept | LangChain | StoneChain |
|---------|-----------|------------|
| Import | 5+ imports | 1-2 imports |
| Setup | Complex config | Just API key |
| Chains | LCEL pipes | Simple `.add()` |
| Memory | BufferMemory class | Built-in |
| Agents | AgentExecutor | Simple `Agent` |
| RAG | VectorStore + Retriever | Simple `RAG` |

## Migration Checklist

- [ ] Replace `langchain_*` imports with `stonechain`
- [ ] Replace `ChatPromptTemplate` with string templates using `{variable}`
- [ ] Replace `ConversationBufferMemory` with `Conversation`
- [ ] Replace `AgentExecutor` with `Agent`
- [ ] Replace vector store RAG with `RAG` class
- [ ] Replace LCEL chains with `Chain.add()`
- [ ] Remove unused dependencies from requirements.txt

## Benefits After Migration

1. **Faster startup** - No more 3-second import times
2. **Simpler debugging** - Read the entire codebase in 30 minutes
3. **Fewer dependencies** - From 200+ to 0
4. **Smaller install** - From 50MB+ to 36KB
5. **Better understanding** - No more abstraction hell

---

Built like a rock. 🪨
