from langgraph.graph import StateGraph, START, END
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresCheckpointer
import os
from dotenv import load_dotenv
import uuid
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import logging
from typing_extensions import NotRequired

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_system_prompt(user: Dict[str, Any], course: Optional[Dict[str, Any]] = None) -> str:
    """Generate the system prompt for the Genoa AI assistant."""
    if course:
        return f"""You are Genoa, an AI teaching assistant for {course.get('name', 'this course')}.
        
        Current Context:
        - You are speaking with {user.get('name', 'a student')}
        - Course: {course.get('name', 'Unknown')}
        - You have access to the course materials and can answer questions about assignments, deadlines, and content
        
        Guidelines:
        - Always be helpful, encouraging, and educational
        - If you don't know something, be honest and offer to find more information
        - Keep responses concise but informative
        - Focus on helping students learn and succeed in their coursework
        """
    else:
        return f"""You are Genoa, an AI teaching assistant.
        
        Current Context:
        - You are speaking with {user.get('name', 'a student')}
        
        Guidelines:
        - Always be helpful, encouraging, and educational
        - If you don't know something, be honest and offer to find more information
        - Keep responses concise but informative
        - Focus on helping students learn and succeed in their coursework
        """

# Define state as a TypedDict
class GenoaState(TypedDict):
    messages: List[Dict[str, Any]]
    user: NotRequired[Dict[str, Any]]
    course: NotRequired[Dict[str, Any]]
    course_id: NotRequired[str]
    next: NotRequired[str]
    canvas_data: NotRequired[Dict[str, Any]]

# Define state reducers
def reduce_messages(current_state: List[Dict[str, Any]], new_state: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return current_state + new_state

def reduce_or_keep(current_state: Any, new_state: Any) -> Any:
    return new_state or current_state

def reduce_dict(current_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
    if current_state and new_state:
        return {**current_state, **new_state}
    return new_state or current_state or {}

def analyze_prompt(state: GenoaState) -> Dict[str, Any]:
    """Analyze the user prompt to understand the intent and required data."""
    try:
        # Get the last message from the user
        last_message = state["messages"][-1]
        if last_message.get("type") != "human":
            return {"next": "generate_response"}
            
        user_prompt = last_message.get("content", "")
        
        # Initialize Gemini model
        model = ChatVertexAI(
            model_name="gemini-1.5-pro",
            temperature=0.2,
            max_output_tokens=1024,
        )
        
        # Create prompt for analysis
        analysis_prompt = f"""
        Analyze the following student query and identify:
        1. The intent (question about assignment, deadline, course content, etc.)
        2. Any specific course elements mentioned (assignment names, deadlines, etc.)
        3. What Canvas data might be needed to answer this query
        
        Student query: {user_prompt}
        
        Provide your analysis in JSON format:
        {{
            "intent": "string",
            "course_elements": ["string"],
            "canvas_data_needed": ["string"],
            "needs_canvas_data": true/false
        }}
        """
        
        # Get analysis from model
        analysis_response = model.invoke(analysis_prompt)
        analysis = analysis_response.content
        
        # Extract JSON from response if needed
        # (Add JSON parsing code here if needed)
        
        return {
            "analysis": analysis,
            "next": "fetch_canvas" if analysis.get("needs_canvas_data", False) else "generate_response"
        }
    except Exception as e:
        logger.error(f"Error in analyze_prompt: {e}")
        return {
            "next": "generate_response",
            "error": str(e)
        }

def fetch_canvas_data(state: GenoaState) -> Dict[str, Any]:
    """Fetch relevant data from Canvas API based on the analysis."""
    try:
        analysis = state.get("analysis", {})
        user_prompt = state["messages"][-1].get("content", "")
        course_id = state.get("course_id")
        
        # Initialize Canvas API connection
        # (Add Canvas API connection code here)
        
        canvas_data = {}
        
        # Determine what data to fetch based on analysis
        data_needed = analysis.get("canvas_data_needed", [])
        
        if "assignments" in data_needed:
            # Fetch assignments from Canvas
            # canvas_data["assignments"] = ...
            pass
            
        if "deadlines" in data_needed:
            # Fetch deadlines from Canvas
            # canvas_data["deadlines"] = ...
            pass
            
        if "course_content" in data_needed:
            # Fetch course content from Canvas
            # canvas_data["course_content"] = ...
            pass
        
        return {
            "canvas_data": canvas_data,
            "next": "generate_response"
        }
    except Exception as e:
        logger.error(f"Error in fetch_canvas_data: {e}")
        return {
            "next": "generate_response",
            "error": str(e)
        }

def generate_response(state: GenoaState) -> Dict[str, Any]:
    """Generate a response using the LLM with all available context."""
    try:
        # Get all required data from state
        messages = state.get("messages", [])
        user = state.get("user", {})
        course = state.get("course", {})
        canvas_data = state.get("canvas_data", {})
        
        # Get the last message
        last_message = messages[-1] if messages else {"content": ""}
        
        # Initialize Gemini model
        model = ChatVertexAI(
            model_name="gemini-1.5-pro",
            temperature=0.7,
        )
        
        # Create agent with tools if needed
        # tools = []
        agent = create_react_agent(model, [])
        
        # Prepare context and messages
        system_prompt = get_system_prompt(user, course)
        system_message = SystemMessage(content=system_prompt)
        
        # Add Canvas data context if available
        canvas_context = ""
        if canvas_data:
            canvas_context = "Canvas data:\n" + str(canvas_data)
            canvas_message = SystemMessage(content=canvas_context)
            context_messages = [system_message, canvas_message]
        else:
            context_messages = [system_message]
        
        # Convert state messages to langchain message format
        conversation_messages = []
        for msg in messages:
            if msg.get("type") == "human":
                conversation_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai":
                conversation_messages.append(AIMessage(content=msg.get("content", "")))
        
        # Get response from agent
        agent_response = agent.invoke({
            "messages": context_messages + conversation_messages
        })
        
        # Format and return the response
        ai_message = {"type": "ai", "content": agent_response.content}
        
        return {
            "messages": [ai_message],
            "next": "FINISH"
        }
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        error_message = {"type": "ai", "content": "I'm sorry, I encountered an error while processing your request. Please try again."}
        return {
            "messages": [error_message],
            "next": "FINISH"
        }

# Create the LangGraph
genoa_graph = StateGraph(GenoaState)

# Add nodes to the graph
genoa_graph.add_node("analyze_prompt", analyze_prompt)
genoa_graph.add_node("fetch_canvas_data", fetch_canvas_data)
genoa_graph.add_node("generate_response", generate_response)

# Add edges between nodes
genoa_graph.set_entry_point("analyze_prompt")
genoa_graph.add_conditional_edges(
    "analyze_prompt",
    lambda x: x["next"],
    {
        "fetch_canvas": "fetch_canvas_data",
        "generate_response": "generate_response"
    }
)
genoa_graph.add_edge("fetch_canvas_data", "generate_response")
genoa_graph.add_edge("generate_response", END)

# Initialize checkpointer for conversation persistence
checkpointer = None
try:
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        checkpointer = PostgresCheckpointer.from_connection_string(os.getenv("POSTGRES_CONNECTION_STRING"))
        logger.info("PostgreSQL checkpointing enabled")
    else:
        logger.info("No PostgreSQL connection string found, running without checkpointing")
except ImportError:
    logger.warning("PostgreSQL checkpointing dependencies not available. To enable checkpointing, install required packages.")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL checkpointing: {e}")

# Compile the graph
compiled_genoa_graph = genoa_graph.compile(checkpointer=checkpointer)

# Function to process a query through the graph
async def process_query(user_prompt: str, course_id: Optional[str] = None, user_info: Optional[Dict[str, Any]] = None):
    """Process a user query through the Genoa LangGraph."""
    try:
        # Initialize state
        initial_state = {
            "messages": [{"type": "human", "content": user_prompt}],
            "course_id": course_id,
            "user": user_info or {},
        }
        
        # If course_id is provided, get course info
        if course_id:
            # Fetch course info from Canvas or database
            # course_info = ...
            initial_state["course"] = {} # Replace with actual course info
        
        # Run the graph
        config = {"configurable": {"userId": user_info.get("id", str(uuid.uuid4()))}} if user_info else {}
        result = await compiled_genoa_graph.ainvoke(initial_state, config=config)
        
        # Extract the AI response from the final state
        ai_messages = [msg for msg in result.get("messages", []) if msg.get("type") == "ai"]
        ai_response = ai_messages[-1].get("content", "") if ai_messages else ""
        
        return {
            "response": ai_response,
            "canvas_data": result.get("canvas_data", {}),
        }
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        return {
            "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
            "error": str(e)
        }

if __name__ == "__main__":
    genoa = GenoaAI()
    response = genoa.process_query("How do I improve my English grade?")
    print("\nProcessed Response:")
    print(response['response'])  
    print("\nRaw Canvas Data:")
    print(response['raw_data'])  

    response2 = genoa.process_query("What is my current grade in CSC 121?")
    print("\nProcessed Response:")
    print(response2['response'])  
    print("\nRaw Canvas Data:")
    print(response2['raw_data'])

    response3 = genoa.process_query("What are my assignments for Freshman English 2?")
    print("\nProcessed Response:")
    print(response3['response'])  
    print("\nRaw Canvas Data:")
    print(response3['raw_data'])
