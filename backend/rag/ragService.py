from ragInit import loadDocs, getSplit, getVectorStore, getRetriever, getChain

### 아래 4개는 처음에 한번만 하면 됌
docs = loadDocs()
splits = getSplit(docs)
vectorstore = getVectorStore(splits)
retriever = getRetriever(vectorstore)


# 사용자의 질문, 쿼리
query = "정규화에 대해 설명해줘줘" # 여긴 사용자 입력으로 바뀌어야함함
result_docs = retriever.invoke(query) # 쿼리 호출하여 retriever로 검색

#### Creating a Prompt with Retrieved Result
chain = getChain()
response = chain.invoke({"context": result_docs, "input": query})
print(response)
### response를 화면에 뿌리며 됌