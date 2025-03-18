import os
from typing import Dict, List, Any, Tuple, Optional, Annotated, Literal, TypedDict, Union
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from Canvas_Context import CanvasManager

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Canvas Manager
canvas_manager = CanvasManager()

# Define the tools
@tool
def get_current_classes() -> List[Dict[str, Any]]:
    """Fetch all current classes for the user from Canvas."""
    return canvas_manager.get_current_classes()

@tool
def get_class_assignments(course_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all assignments for a specific class.
    
    Args:
        course_id: The Canvas course ID
    """
    return canvas_manager.get_class_assignments(course_id)

@tool
def get_class_syllabus(course_id: str) -> Dict[str, str]:
    """Fetch syllabus for a specific class.
    
    Args:
        course_id: The Canvas course ID
    """
    return canvas_manager.get_class_syllabus(course_id)

@tool
def get_class_grades(course_id: str) -> Dict[str, Any]:
    """Fetch grades for a specific class.
    
    Args:
        course_id: The Canvas course ID
    """
    return canvas_manager.get_class_grades(course_id)

@tool
def get_upcoming_tests(course_id: str) -> List[Dict[str, Any]]:
    """Fetch upcoming tests/quizzes for a specific class.
    
    Args:
        course_id: The Canvas course ID
    """
    return canvas_manager.get_upcoming_tests(course_id)

@tool
def find_course_id(class_name: str) -> str:
    """Find course ID for a given class name.
    
    Args:
        class_name: The name of the class to search for
    """
    return canvas_manager.find_course_id(class_name)

# Create a list of all available tools
tools = [
    get_current_classes,
    get_class_assignments,
    get_class_syllabus,
    get_class_grades,
    get_upcoming_tests,
    find_course_id
]

# Define the state schema
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    next_steps: List[str]

# Define the tool selection model
class ToolSelection(BaseModel):
    tools: List[str] = Field(description="List of tool names to use")
    reasoning: str = Field(description="Reasoning for selecting these tools")

# Create the Gemini model with retry logic for quota errors
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(retry=retry_if_exception_type(ResourceExhausted),
       wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def create_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        convert_system_message_to_human=True,
        max_output_tokens=1024  # Limiting token output to reduce API usage
    )

try:
    model = create_gemini_model()
except ResourceExhausted:
    # Fallback to a simpler model if quota is exhausted
    print("Warning: API quota exceeded for gemini-1.5-pro. Falling back to gemini-pro model.")
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True,
        max_output_tokens=512  # Further limiting token output
    )

# Create the tool selection prompt
tool_selection_prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="""You are an assistant that helps users with their Canvas LMS tasks.
Your job is to analyze the user's query and select the appropriate tools to use.
Available tools:
- get_current_classes: Fetch all current classes for the user
- get_class_assignments: Fetch all assignments for a specific class
- get_class_syllabus: Fetch syllabus for a specific class
- get_class_grades: Fetch grades for a specific class
- get_upcoming_tests: Fetch upcoming tests/quizzes for a specific class
- find_course_id: Find course ID for a given class name

Guidelines for tool selection:
1. Select only the tools that are necessary to answer the user's query.
2. If the user mentions a specific course or class, ALWAYS select find_course_id first.
3. If the user asks about grades, select get_class_grades.
4. If the user asks about assignments, select get_class_assignments.
5. If the user asks about tests or quizzes, select get_upcoming_tests.
6. If the user asks about the syllabus, select get_class_syllabus.
7. If the user doesn't mention a specific class, select get_current_classes.

Examples:
- "What's my grade in Math?" → ["find_course_id", "get_class_grades"]
- "Show me my assignments for Biology" → ["find_course_id", "get_class_assignments"]
- "What classes am I taking?" → ["get_current_classes"]
"""),
    MessagesPlaceholder(variable_name="messages"),
    HumanMessage(content="Select the tools to use for this query. Return your selection as a JSON object with a list of tool names.")
])

# Create the tool execution prompt
tool_execution_prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="""You are an assistant that helps users with their Canvas LMS tasks.
You have access to several tools to help answer the user's questions.
Use the tools to gather information, then provide a helpful response.
"""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="tool_results"),
    HumanMessage(content="Based on the tool results, provide a helpful response to the user's query.")
])

# Define the functions for the graph
def select_tools(state: AgentState) -> AgentState:
    """Select which tools to use based on the user's query."""
    # Create a chain for tool selection
    tool_selection_chain = tool_selection_prompt | model | JsonOutputParser()
    
    # Get the tool selection
    tool_selection = tool_selection_chain.invoke({"messages": state["messages"]})
    
    # Update the state with the next steps
    state["next_steps"] = tool_selection.get("tools", [])
    
    return state

def execute_tools(state: AgentState) -> AgentState:
    """Execute the selected tools."""
    tool_results = []
    tool_calls = []
    
    # Get the last user message
    last_user_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    
    if not last_user_message:
        return state
        
    # Print the user query for debugging
    print(f"Processing user query: '{last_user_message.content}'")
    
    # Create a mapping of tool names to tool functions
    tool_map = {
        "get_current_classes": get_current_classes,
        "get_class_assignments": get_class_assignments,
        "get_class_syllabus": get_class_syllabus,
        "get_class_grades": get_class_grades,
        "get_upcoming_tests": get_upcoming_tests,
        "find_course_id": find_course_id
    }
    
    # Execute each tool
    for tool_name in state["next_steps"]:
        # Find the tool
        tool_fn = tool_map.get(tool_name)
        
        if not tool_fn:
            continue
            
        # Check if the tool requires parameters
        if tool_name in ["get_class_assignments", "get_class_syllabus", "get_class_grades", "get_upcoming_tests"]:
            # If we need a course_id, try to find it
            # Extract potential course names from the query
            # Common course subjects and names to look for
            course_keywords = ["math", "biology", "chemistry", "physics", "history", "english", 
                            "science", "algebra", "calculus", "discrete", "statistics", 
                            "computer", "programming", "art", "music", "economics", 
                            "psychology", "sociology", "philosophy"]
            
            course_name = None
            message_words = last_user_message.content.lower().split()
            
            # First look for common course name patterns
            for i, word in enumerate(message_words):
                # Check for explicit course mentions
                if word in ["course", "class", "subject"] and i < len(message_words) - 1:
                    course_name = message_words[i+1]
                    break
                # Check for course keywords
                if any(keyword in word for keyword in course_keywords):
                    course_name = word
                    break
                # Check for two-word course names (e.g., "discrete math")
                if i < len(message_words) - 1:
                    two_word = f"{word} {message_words[i+1]}"
                    if any(keyword in two_word for keyword in course_keywords):
                        course_name = two_word
                        break
            
            if course_name:
                # First get the course_id
                course_id = None
                try:
                    course_id = find_course_id.func(class_name=course_name)
                except Exception as e:
                    print(f"Error finding course ID: {e}")
                
                if course_id:
                    try:
                        # Call the tool function directly with the right parameters
                        if tool_name == "get_class_assignments":
                            result = get_class_assignments.func(course_id=course_id)
                        elif tool_name == "get_class_syllabus":
                            result = get_class_syllabus.func(course_id=course_id)
                        elif tool_name == "get_class_grades":
                            result = get_class_grades.func(course_id=course_id)
                        elif tool_name == "get_upcoming_tests":
                            result = get_upcoming_tests.func(course_id=course_id)
                            
                        tool_calls.append({"tool": tool_name, "args": {"course_id": course_id}})
                        tool_results.append({"tool": tool_name, "result": result})
                    except Exception as e:
                        print(f"Error executing tool {tool_name}: {e}")
        elif tool_name == "find_course_id":
            # Extract potential course names from the query
            # Common course subjects and names to look for
            course_keywords = ["math", "biology", "chemistry", "physics", "history", "english", 
                            "science", "algebra", "calculus", "discrete", "statistics", 
                            "computer", "programming", "art", "music", "economics", 
                            "psychology", "sociology", "philosophy"]
            
            course_name = None
            message_words = last_user_message.content.lower().split()
            
            # First look for common course name patterns
            for i, word in enumerate(message_words):
                # Check for explicit course mentions
                if word in ["course", "class", "subject"] and i < len(message_words) - 1:
                    course_name = message_words[i+1]
                    break
                # Check for course keywords
                if any(keyword in word for keyword in course_keywords):
                    course_name = word
                    break
                # Check for two-word course names (e.g., "discrete math")
                if i < len(message_words) - 1:
                    two_word = f"{word} {message_words[i+1]}"
                    if any(keyword in two_word for keyword in course_keywords):
                        course_name = two_word
                        break
            
            if course_name:
                try:
                    result = find_course_id.func(class_name=course_name)
                    tool_calls.append({"tool": tool_name, "args": {"class_name": course_name}})
                    tool_results.append({"tool": tool_name, "result": result})
                except Exception as e:
                    print(f"Error executing find_course_id: {e}")
        else:
            # Tool doesn't require parameters
            try:
                result = get_current_classes.func()
                tool_calls.append({"tool": tool_name, "args": {}})
                tool_results.append({"tool": tool_name, "result": result})
            except Exception as e:
                print(f"Error executing {tool_name}: {e}")
    
    # Update the state
    state["tool_calls"] = tool_calls
    state["tool_results"] = tool_results
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the tool results."""
    # Create a chain for response generation
    response_chain = tool_execution_prompt | model
    
    # Format the tool results as messages
    tool_result_messages = []
    for result in state["tool_results"]:
        tool_result_messages.append(
            AIMessage(content=f"Tool: {result['tool']}\nResult: {result['result']}")
        )
    
    # Generate the response
    response = response_chain.invoke({
        "messages": state["messages"],
        "tool_results": tool_result_messages
    })
    
    # Add the response to the messages
    state["messages"].append(response)
    
    return state

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if we should continue or end the process."""
    # If we have tool results, we're done
    if state["tool_results"]:
        return "end"
    return "continue"

# Create the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("select_tools", select_tools)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("generate_response", generate_response)

# Add the edges
workflow.add_edge("select_tools", "execute_tools")
workflow.add_edge("execute_tools", "generate_response")
workflow.add_edge("generate_response", END)

# Set the entry point
workflow.set_entry_point("select_tools")

# Compile the graph
agent = workflow.compile()

def process_user_query(query: str) -> str:
    """Process a user query and return a response."""
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=query)],
        "tool_calls": [],
        "tool_results": [],
        "next_steps": []
    }
    
    try:
        # Run the agent
        result = agent.invoke(state)
        
        # Return the last message
        return result["messages"][-1].content
    except ResourceExhausted:
        return "I'm sorry, but I've hit the API rate limit. Please try again in a few minutes or ask a simpler question."
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your request. Please try again with a different question or check your API key configuration."

# Example usage
if __name__ == "__main__":
    # Test the agent
    query = input("Enter your Canvas query: ")
    response = process_user_query(query)
    print("\nAgent Response:")
    print(response)
