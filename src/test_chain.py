from injector import Injector, Module, provider
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableLambda

from src.chain import WebRetrievalChain


class FakeModule(Module):
    @provider
    def model(self) -> BaseChatModel:
        responses: list[BaseMessage] = [AIMessage("fake output")]
        return FakeMessagesListChatModel(responses=responses)

    @provider
    def retriever(self) -> RetrieverLike:
        return RunnableLambda(lambda _: [Document("fake document")])


def test_web_retrieval_chain() -> None:
    injector = Injector([FakeModule()])
    chain = injector.get(WebRetrievalChain)
    output = chain.invoke("東京の明日の天気は？")
    assert output == "fake output"
