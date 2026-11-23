from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

## TOOLS
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import requests
import os
import math

## RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# HITL
from langgraph.types import interrupt, Command


load_dotenv()  

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",max_output_tokens=500)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#-------------- PDF Loading and all indside function------------

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    return len(chunks)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever
# -------------------------------------------------------------
#------------------------ Tools -------------------------------

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math":math,
            "abs":abs,
            "round":round,
            "min":min,
            "max":max,
            "sum":sum
        }

        result = eval(expression, {"__builtins__": {}},allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL','TSLA')
    using Alpha Vantage with API key in the URL.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given qunatity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes"/anything else).
    """

    # This pause the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower()=="yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol":symbol,
            "quantity":quantity,
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol":symbol,
            "quantity":quantity,   
        }

@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a given city.
    Returns temperature, feels-like temperature,
    humidity, wind speed and weather conditions.
    """

    try:
        # Step 1: Get city coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )
        geo_response.raise_for_status()

        locations = geo_response.json().get("results", [])

        if not locations:
            return f"City not found: {city}"

        location = locations[0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        city_name = location["name"]

        # Step 2: Fetch current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto"
            },
            timeout=15
        )
        weather_response.raise_for_status()

        data = weather_response.json()
        current = data["current"]

        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snowfall",
            73: "Moderate snowfall",
            75: "Heavy snowfall",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm"
        }

        condition = weather_codes.get(
            current["weather_code"],
            "Unknown weather condition"
        )

        return f"""
Current Weather: {city_name}

Temperature: {current['temperature_2m']} °C
Feels Like: {current['apparent_temperature']} °C
Condition: {condition}
Humidity: {current['relative_humidity_2m']}%
Wind Speed: {current['wind_speed_10m']} km/h
Precipitation: {current['precipitation']} mm
Updated At: {current['time']}
"""

    except requests.RequestException as e:
        return f"Weather API error: {str(e)}"

    except Exception as e:
        return f"Weather tool error: {str(e)}"


## ------------------------ RAG Tool ------------------
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use the tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    try:
        retriever = get_retriever()          # <-- CHANGED: load it here
    except Exception:
        return "No PDF has been uploaded yet. Ask the user to upload a PDF first."

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page","Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)
# -------------------------------------------------------------

# ------------------------------ Tool Bind --------------------

tools = [calculator, search_tool, get_stock_price, get_weather,rag_tool,purchase_stock]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)
# -------------------------------------------------------------

# ------------------------------ Chatbot State Graph ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------------------------------------


def chat_node(state: ChatState):

    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Also Your name whenever anyone asks say SurjyaAI.\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state['messages']
    ]

    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}

conn = sqlite3.connect(database="chatbot_state.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

## add Nodes

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)


## add edges

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()

    for check in checkpoint.list(None):
        all_threads.add(check.config['configurable']['thread_id'])

    return list(all_threads)


if __name__ == "__main__":
    print("Agnetic Chatbot CLI\n")
    print("Type 'exit' to quit\n")

    thread_id = "demo-1234"

    while True:
        user_input = input("You:")
        if user_input.lower().strip() in {"exit","quit"}:
            print("Goodbye")
            break

        state = {"messages":[HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable":{"thread_id":thread_id}},
        )

        interrupts = result.get("__interrupt__",[])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your Decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}}
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.text}\n")

