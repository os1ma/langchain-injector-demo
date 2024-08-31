from dotenv import load_dotenv
from injector import Injector, Module, inject, provider
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


class RetrievalChain:
    @inject
    def __init__(self, model: BaseChatModel, retriever: RetrieverLike):
        self.model = model
        self.retriever = retriever

    def invoke(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer the question.\n\nContext: {context}"),
                ("human", "{question}"),
            ],
        )

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )

        return chain.invoke(question)


class ProdModule(Module):
    @provider
    def model(self) -> BaseChatModel:
        return ChatOpenAI(model="gpt-4o-mini")

    @provider
    def retriever(self) -> RetrieverLike:
        return TavilySearchAPIRetriever(k=3)


def main() -> None:
    injector = Injector([ProdModule()])
    chain = injector.get(RetrievalChain)
    output = chain.invoke("東京の明日の天気は？")
    print(output)


if __name__ == "__main__":
    main()
