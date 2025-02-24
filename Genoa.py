import os
from dotenv import load_dotenv
import google.generativeai as genai
from Canvas_Context import CanvasManager

class GenoaAI:
    def __init__(self):
        # Load environment variables and initialize Gemini
        load_dotenv()
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        self.canvas = CanvasManager()

    def analyze_prompt(self, user_prompt, course_id=None):
        """Analyze user prompt to determine which Canvas data to fetch"""
        system_prompt = """
        You are an AI tutor that analyzes user queries about their Canvas course information.
        Based on the query, determine which Canvas functions are needed from these options in order to answer the user's question best:
        - get_current_classes(): Get list of all current classes
        - get_class_assignments(course_id): Get all assignments for a class
        - get_class_syllabus(course_id): Get the class syllabus
        - get_class_grades(course_id): Get grades for a class
        - get_upcoming_tests(course_id): Get upcoming tests/quizzes
        
        Respond with a JSON-like structure containing:
        - functions: list of required function names
        
        Example response:
        {
            "functions": ["get_current_classes"]
        }
        """

        prompt = f"{system_prompt}\n\nUser query: {user_prompt}"
        
        try:
            response = self.model.generate_content(prompt)
            analysis = response.text
            
            # Execute the determined Canvas functions
            return self._fetch_canvas_data(analysis, course_id)
            
        except Exception as e:
            return f"Error analyzing prompt: {str(e)}"

    def _fetch_canvas_data(self, analysis, course_id):
        """Fetch the required Canvas data based on the AI analysis"""
        results = {}
        
        if "get_current_classes" in analysis:
            results['classes'] = self.canvas.get_current_classes()
            
        if course_id:
            if "get_class_assignments" in analysis:
                results['assignments'] = self.canvas.get_class_assignments(course_id)
                
            if "get_class_syllabus" in analysis:
                results['syllabus'] = self.canvas.get_class_syllabus(course_id)
                
            if "get_class_grades" in analysis:
                results['grades'] = self.canvas.get_class_grades(course_id)
                
            if "get_upcoming_tests" in analysis:
                results['upcoming_tests'] = self.canvas.get_upcoming_tests(course_id)
        
        return results

    def find_course_id(self, query):
        courses = self.canvas.get_current_classes()
        if not courses:
            return None
        
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        for course in courses:
            if any(keyword in course['course_name'].lower() for keyword in query.split()):
                return course['course_id']
        return None

    def process_query(self, user_prompt, course_id=None):
        if not course_id:
            course_id = self.find_course_id(user_prompt)
        """Main method to process user queries"""
        try:
            # First, analyze the prompt and get Canvas data
            canvas_data = self.analyze_prompt(user_prompt, course_id)
            
            # Create a detailed system prompt that helps the AI understand how to use the Canvas data
            system_prompt = """
            You are a helpful academic assistant with access to Canvas course information.
            Please provide a clear and specific answer based on the available Canvas data.
            If the data doesn't contain enough information to fully answer the question,
            acknowledge this and answer with what is available.
            
            Format numbers, dates, and important information clearly.
            If listing multiple items, use bullet points for clarity.
            """
            
            # Construct a detailed context with both the data and instructions
            context = f"""{system_prompt}

            Available Canvas Data:
            {str(canvas_data)}

            User Question: {user_prompt}

            Please provide a helpful response using this information."""
            
            # Generate the response using the enhanced context
            response = self.model.generate_content(context)
            
            return {
                'raw_data': canvas_data,
                'response': response.text
            }
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

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
