import os
from dotenv import load_dotenv
from canvasapi import Canvas
from datetime import datetime

class CanvasManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Canvas connection
        self.canvas = Canvas(os.getenv('CANVAS_URL'), os.getenv('Canvas_API_KEY'))
        self.user = self.canvas.get_current_user()

    def get_current_classes(self):
        """Fetch all current classes for the user"""
        try:
            courses = self.user.get_courses()
            current_year = str(datetime.now().year)
            
            # Determine current term based on month
            month = datetime.now().month
            if 1 <= month <= 5:
                current_term = "SP"  
            elif 6 <= month <= 7:
                current_term = "SU"  
            else:
                current_term = "FA"  
            
            term_prefix = f"{current_year}{current_term}"
            
            return [{
                'course_name': course.name,
                'course_id': course.id
            } for course in courses if term_prefix in course.name]
        except Exception as e:
            print(f"Error fetching classes: {str(e)}")
            return None

    def get_class_assignments(self, course_id):
        """Fetch all assignments for a specific class"""
        try:
            course = self.canvas.get_course(course_id)
            now = datetime.now()
            assignments = {
                'upcoming': [],
                'past': [],
                'missing': []
            }
            
            for assignment in course.get_assignments():
                assignment_data = {
                    'name': assignment.name,
                    'id': assignment.id,
                    'due_date': getattr(assignment, 'due_at', None),
                    'description': getattr(assignment, 'description', ''),
                    'points_possible': getattr(assignment, 'points_possible', None)
                }
                
                try:
                    submission = assignment.get_submission(self.user.id)
                    assignment_data.update({
                        'submission_status': submission.workflow_state,
                        'score': submission.score,
                        'submitted_at': getattr(submission, 'submitted_at', None),
                        'late': getattr(submission, 'late', False)
                    })
                except Exception:
                    pass

                if assignment.due_at:
                    due_date = datetime.strptime(assignment.due_at, "%Y-%m-%dT%H:%M:%SZ")
                    if due_date > now:
                        assignments['upcoming'].append(assignment_data)
                    elif assignment_data.get('submission_status') in ['submitted', 'graded']:
                        assignments['past'].append(assignment_data)
                    else:
                        assignments['missing'].append(assignment_data)
                else:
                    assignments['upcoming'].append(assignment_data)
                    
            return assignments
        except Exception as e:
            print(f"Error fetching assignments: {str(e)}")
            return None

    def get_class_syllabus(self, course_id):
        """Fetch syllabus for a specific class"""
        try:
            course = self.canvas.get_course(course_id)
            return {
                'syllabus_body': course.syllabus_body,
                'course_name': course.name
            }
        except Exception as e:
            print(f"Error fetching syllabus: {str(e)}")
            return None

    def get_class_grades(self, course_id):
        """Fetch grades for a specific class"""
        try:
            course = self.canvas.get_course(course_id)
            enrollments = course.get_enrollments()
            for enrollment in enrollments:
                if enrollment.user_id == self.user.id:
                    grades = getattr(enrollment, 'grades', {})
                    return {
                        'current_score': grades.get('current_score'),
                        'final_score': grades.get('final_score'),
                        'current_grade': grades.get('current_grade'),
                        'final_grade': grades.get('final_grade')
                    }
            return None
        except Exception as e:
            print(f"Error fetching grades: {str(e)}")
            return None

    def get_upcoming_tests(self, course_id):
        """Fetch upcoming tests/quizzes for a specific class"""
        try:
            course = self.canvas.get_course(course_id)
            now = datetime.now()
            upcoming_tests = []
            
            for assignment in course.get_assignments():
                if any(term in assignment.name.lower() for term in ['test', 'quiz', 'exam', 'midterm', 'final']):
                    if assignment.due_at:
                        due_date = datetime.strptime(assignment.due_at, "%Y-%m-%dT%H:%M:%SZ")
                        if due_date > now:
                            upcoming_tests.append({
                                'name': assignment.name,
                                'due_date': assignment.due_at,
                                'points_possible': getattr(assignment, 'points_possible', None),
                                'description': getattr(assignment, 'description', '')
                            })
            
            return upcoming_tests
        except Exception as e:
            print(f"Error fetching upcoming tests: {str(e)}")
            return None


    def find_course_id(self, class_name):
        """Find course ID for a given class name"""
        try:
            courses = self.canvas.get_courses()
            for course in courses:
                if class_name.lower() in course.name.lower():
                    return course.id
            return None
        except Exception as e:
            print(f"Error finding course ID: {str(e)}")
            return None
